use crate::eval::lbf_evaluator::LBFEvaluator;
use crate::eval::sample_eval::SampleEval;
use crate::sample::search::{search_placement, SampleConfig};
use itertools::Itertools;
use jagua_rs::geometry::DTransformation;
use jagua_rs::probs::spp::entities::{SPInstance, SPPlacement, SPProblem};
use jagua_rs::Instant;
use log::debug;
use ordered_float::OrderedFloat;
use std::cmp::Reverse;
use std::iter;
use rand::rngs::Xoshiro256PlusPlus;

/// The initial-placement heuristic could not construct a solution.
/// This is not a proof that the instance is infeasible.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConstructionError {
    /// The item's ID in the input instance.
    pub external_id: u64,
}

impl std::fmt::Display for ConstructionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "could not construct an initial placement for item {}", self.external_id)
    }
}

impl std::error::Error for ConstructionError {}

pub struct LBFBuilder {
    pub instance: SPInstance,
    pub prob: SPProblem,
    pub rng: Xoshiro256PlusPlus,
    pub sample_config: SampleConfig,
    /// Fork: items pinned at a fixed transform. Placed first during `construct`
    /// and never entered into the LBF search order.
    pub frozen: Vec<(usize, DTransformation)>,
    /// Item idxs `>= frozen_threshold` are frozen. `usize::MAX` = none.
    pub frozen_threshold: usize,
}

impl LBFBuilder {
    pub fn new(
        instance: SPInstance,
        rng: Xoshiro256PlusPlus,
        sample_config: SampleConfig,
        frozen: Vec<(usize, DTransformation)>,
        frozen_threshold: usize,
    ) -> anyhow::Result<Self> {
        let prob = SPProblem::new(instance.clone())?;

        Ok(Self {
            instance,
            prob,
            rng,
            sample_config,
            frozen,
            frozen_threshold,
        })
    }

    /// Builds a complete initial placement.
    ///
    /// Returns an error if strip growth reaches the heuristic limit.
    pub fn construct(mut self) -> anyhow::Result<Self> {
        let start = Instant::now();

        // Fork: place the pinned (frozen) items first at their fixed transforms.
        for &(item_idx, d_transf) in &self.frozen {
            self.prob.place_item(SPPlacement { item_idx, d_transf });
        }

        let n_items = self.instance.items.len();
        let sorted_item_indices = (0..n_items)
            .filter(|id| *id < self.frozen_threshold)
            .sorted_by_cached_key(|id| {
                let item_shape = self.instance.item(*id).shape_cd().as_ref();
                let convex_hull_area = item_shape.surrogate().convex_hull_area;
                let diameter = item_shape.diameter();
                Reverse(OrderedFloat(convex_hull_area * diameter))
            })
            .flat_map(|id| {
                let missing_qty = self.prob.item_demand_qtys()[id];
                iter::repeat_n(id, missing_qty)
            })
            .collect_vec();

        debug!("[CONSTR] placing items in order: {:?}",sorted_item_indices);

        for item_idx in sorted_item_indices {
            self.place_item(item_idx)?;
        }

        self.prob.fit_strip()?;
        debug!("[CONSTR] placed all items in width: {:.3} (in {:?})",self.prob.strip_width(), start.elapsed());
        Ok(self)
    }

    /// Fork: a bounded constructive pass. Each copy is visited once; a copy
    /// that cannot be placed stays unplaced. No exploration, compression or
    /// separation is run. With `max_width` the strip is never widened.
    pub fn construct_once(
        mut self,
        max_width: Option<f32>,
        cancelled: impl Fn() -> bool,
    ) -> anyhow::Result<Self> {
        if let Some(width) = max_width {
            self.prob.change_strip_width(width)?;
        }
        for &(item_idx, d_transf) in &self.frozen {
            self.prob.place_item(SPPlacement { item_idx, d_transf });
        }
        let order = (0..self.instance.items.len())
            .filter(|id| *id < self.frozen_threshold)
            .sorted_by_cached_key(|id| {
                let shape = self.instance.item(*id).shape_cd().as_ref();
                Reverse(OrderedFloat(shape.surrogate().convex_hull_area * shape.diameter()))
            })
            .flat_map(|id| iter::repeat_n(id, self.prob.item_demand_qtys()[id]))
            .collect_vec();
        // Same heuristic ceiling as `place_item`, without treating it as infeasibility.
        let width_limit = 2.0 * self.instance.items.iter()
            .map(|(item, qty)| item.shape_cd().diameter() * *qty as f32)
            .sum::<f32>();
        for item_idx in order {
            // Expansion only makes room for this copy. The bounded search can
            // fail (e.g. a shape cannot fit the strip) without losing others.
            for _ in 0..32 {
                if cancelled() {
                    return Ok(self);
                }
                if let Some(p) = self.find_placement(item_idx) {
                    self.prob.place_item(p);
                    break;
                }
                let next_width = self.prob.strip_width() * 1.2;
                if max_width.is_some() || next_width >= width_limit {
                    break;
                }
                self.prob.change_strip_width(next_width)?;
            }
        }
        if !self.prob.layout().placed_items().is_empty() {
            self.prob.fit_strip()?;
        }
        Ok(self)
    }

    fn place_item(&mut self, item_idx: usize) -> anyhow::Result<()> {
        loop {
            if let Some(placement) = self.find_placement(item_idx) {
                self.prob.place_item(placement);
                debug!("[CONSTR] placing item {}/{} with idx {} at [{}]", self.prob.layout().placed_items().len(), self.instance.total_item_qty(), placement.item_idx, placement.d_transf);
                return Ok(());
            }

            let next_width = self.prob.strip_width() * 1.2;
            // Retain the existing heuristic ceiling, without treating it as infeasibility.
            let width_limit = 2.0 * self.instance.items.iter()
                .map(|(item, qty)| item.shape_cd().diameter() * *qty as f32)
                .sum::<f32>();
            if next_width >= width_limit {
                let external_id = self.instance.item(item_idx).external_id();
                return Err(ConstructionError { external_id }.into());
            }
            debug!("[CONSTR] failed to place item with idx {}, expanding strip width", item_idx);
            self.prob.change_strip_width(next_width)?;
        }
    }

    fn find_placement(&mut self, item_idx: usize) -> Option<SPPlacement> {
        let layout = self.prob.layout();
        let item = self.instance.item(item_idx);
        let evaluator = LBFEvaluator::new(layout, item);

        let (best_sample, _) = search_placement(layout, item, None, evaluator, self.sample_config, &mut self.rng);

        match best_sample {
            Some((d_transf, SampleEval::Clear { .. })) => {
                Some(SPPlacement { item_idx, d_transf })
            }
            _ => None
        }
    }
}
