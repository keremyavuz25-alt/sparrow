use crate::eval::sep_evaluator::SeparationEvaluator;
use crate::quantify::tracker::CollisionTracker;
use crate::sample::search;
use crate::sample::search::SampleConfig;
use crate::util::assertions::tracker_matches_layout;
use crate::FMT;
use itertools::Itertools;
use jagua_rs::entities::PItemKey;
use jagua_rs::geometry::DTransformation;
use jagua_rs::probs::spp::entities::{SPPlacement, SPProblem, SPSolution};
use log::debug;
use rand::prelude::{IndexedRandom, SliceRandom};
use rand::RngExt;
use crate::sample::uniform_sampler::convert_sample_to_closest_feasible;
use std::iter::Sum;
use std::ops::AddAssign;
use rand::rngs::Xoshiro256PlusPlus;
use tap::Tap;

pub struct SeparatorWorker {
    pub prob: SPProblem,
    pub ct: CollisionTracker,
    pub rng: Xoshiro256PlusPlus,
    pub sample_config: SampleConfig,
    /// Fork: item idxs `>= frozen_item_id_threshold` are pinned and never moved.
    pub frozen_item_id_threshold: usize,
    /// Fork experiment: swap move (see `SeparatorConfig::swap_prob`).
    pub swap_prob: f32,
    pub swap_tau: usize,
    /// Consecutive non-improving separator iterations (set by the master each round).
    pub stagnation: usize,
    /// Swaps kept since the worker was built (diagnostics).
    pub n_swaps: usize,
}

/// Two items are "similar-sized" when their diameters are within this ratio.
const SWAP_SIMILARITY: f32 = 1.3;

impl SeparatorWorker {
    pub fn load(&mut self, sol: &SPSolution, ct: &CollisionTracker) {
        // restores the state of the worker to the given solution and accompanying tracker
        debug_assert!(sol.strip_width() == self.prob.strip_width());
        self.prob.restore(sol);
        self.ct = ct.clone();
    }

    /// Algorithm 5 from https://doi.org/10.48550/arXiv.2509.13329
    pub fn move_items(&mut self) -> SepStats {
        // Collect all colliding items in a random order. Frozen items are
        // excluded (fork): pinned, never selected for a move.
        let candidates = self.prob.layout().placed_items().iter()
            .filter(|(pk, pi)| pi.item().idx() < self.frozen_item_id_threshold && self.ct.get_loss(*pk) > 0.0)
            .map(|(pk, _)| pk)
            .collect_vec()
            .tap_mut(|v| v.shuffle(&mut self.rng));

        let mut total_moves = 0;
        let mut total_evals = 0;

        let swap_p = if self.swap_tau > 0 {
            self.swap_prob * (self.stagnation as f32 / self.swap_tau as f32).min(1.0)
        } else {
            self.swap_prob
        };

        // Give each colliding item the opportunity to move to a better (eval) position
        for &pk in candidates.iter() {
            // A swap earlier in this pass may have re-keyed this item: skip it.
            if !self.prob.layout().placed_items().contains_key(pk) {
                continue;
            }
            // First check if the item is still colliding
            if self.ct.get_loss(pk) > 0.0 {
                let pk = if swap_p > 0.0 && self.rng.random::<f32>() < swap_p {
                    // Even a reverted swap re-keys the item: always use the returned key.
                    let (pk_now, kept) = self.try_swap(pk);
                    if kept {
                        self.n_swaps += 1;
                        if self.ct.get_loss(pk_now) == 0.0 {
                            continue;
                        }
                    }
                    pk_now
                } else {
                    pk
                };
                let item = self.prob.layout().placed_items()[pk].item().clone();
                let item = item.as_ref();

                // Create an 'evaluator' to perform collision detection and collision quantification of the samples during the search
                let evaluator = SeparationEvaluator::new(self.prob.layout(), item, pk, &self.ct, (self.sample_config.clear_x, self.sample_config.clear_y, self.sample_config.clear_contact));

                // Perform the search for a better position for the item
                let (best_sample, n_evals) =
                    search::search_placement(self.prob.layout(), item, Some(pk), evaluator, self.sample_config, &mut self.rng);

                let (new_dt, _eval) = best_sample.expect("search_placement should always return a sample");

                // Move the item to the new position
                self.move_item(pk, new_dt);
                total_moves += 1;
                total_evals += n_evals;
            }
        }
        SepStats { total_moves, total_evals }
    }

    /// Fork experiment: exchange the positions of `pk_a` and a random similar-sized
    /// movable item (rotations snapped to each item's feasible set). Returns item
    /// A's current key (it changes even when the swap is reverted) and whether
    /// the swap was kept.
    fn try_swap(&mut self, pk_a: PItemKey) -> (PItemKey, bool) {
        let (item_a, dt_a, diam_a, area_a) = {
            let pi = &self.prob.layout().placed_items()[pk_a];
            (pi.item().clone(), pi.d_transf(), pi.shape().diameter(), pi.shape().area())
        };
        let frozen = self.frozen_item_id_threshold;
        let cands = self.prob.layout().placed_items().iter()
            .filter(|(pk, pi)| *pk != pk_a && pi.item().idx() != item_a.idx() && pi.item().idx() < frozen)
            .filter(|(_, pi)| {
                let rd = pi.shape().diameter() / diam_a;
                let ra = pi.shape().area() / area_a;
                rd > 1.0 / SWAP_SIMILARITY && rd < SWAP_SIMILARITY
                    && ra > 1.0 / (SWAP_SIMILARITY * SWAP_SIMILARITY) && ra < SWAP_SIMILARITY * SWAP_SIMILARITY
            })
            .map(|(pk, pi)| (pk, pi.item().clone(), pi.d_transf()))
            .collect_vec();
        let Some((pk_b, item_b, dt_b)) = cands.choose(&mut self.rng).cloned() else {
            return (pk_a, false);
        };

        let w_before = self.ct.get_total_weighted_loss();
        let new_dt_a = convert_sample_to_closest_feasible(dt_b, &item_a);
        let new_dt_b = convert_sample_to_closest_feasible(dt_a, &item_b);
        let pk_a2 = self.relocate(pk_a, item_a.idx(), new_dt_a);
        let pk_b2 = self.relocate(pk_b, item_b.idx(), new_dt_b);
        let w_after = self.ct.get_total_weighted_loss();

        if self.swap_tau > 0 || w_after < w_before {
            debug!("Swapped items {} <-> {}: wl {} -> {}", item_a.idx(), item_b.idx(), FMT().fmt2(w_before), FMT().fmt2(w_after));
            (pk_a2, true)
        } else {
            self.relocate(pk_b2, item_b.idx(), dt_b);
            (self.relocate(pk_a2, item_a.idx(), dt_a), false)
        }
    }

    fn relocate(&mut self, pk: PItemKey, item_idx: usize, d_transf: DTransformation) -> PItemKey {
        self.prob.remove_item(pk);
        let new_pk = self.prob.place_item(SPPlacement { d_transf, item_idx });
        self.ct.register_item_move(self.prob.layout(), pk, new_pk);
        new_pk
    }

    pub fn move_item(&mut self, pk: PItemKey, d_transf: DTransformation) -> PItemKey {
        debug_assert!(tracker_matches_layout(&self.ct, self.prob.layout()));

        let (old_l, old_w_l) = (self.ct.get_loss(pk), self.ct.get_weighted_loss(pk));

        debug_assert!(old_l > 0.0, "Item with key {:?} should be colliding, but has no loss: {}", pk, FMT().fmt2(old_l));
        debug_assert!(old_w_l > 0.0, "Item with key {:?} should be colliding, but has no weighted loss: {}", pk, FMT().fmt2(old_w_l));

        // First removing the item and subsequently place it in its new position
        let old_placement = self.prob.remove_item(pk);
        let new_placement = SPPlacement { d_transf, item_idx: old_placement.item_idx };
        let new_pk = self.prob.place_item(new_placement);

        // Update the collision tracker to reflect the changes
        self.ct.register_item_move(self.prob.layout(), pk, new_pk);

        let (new_l, new_w_l) = (self.ct.get_loss(new_pk), self.ct.get_weighted_loss(new_pk));

        debug!("Moved {:?} (l: {}, wl: {}) to {:?} (l+1: {}, wl+1: {})", old_placement, FMT().fmt2(old_l), FMT().fmt2(old_w_l), new_placement, FMT().fmt2(new_l), FMT().fmt2(new_w_l));
        debug_assert!(new_w_l <= old_w_l * 1.001, "weighted loss should never increase: {} > {}", FMT().fmt2(old_w_l), FMT().fmt2(new_w_l));
        debug_assert!(tracker_matches_layout(&self.ct, self.prob.layout()));

        new_pk
    }
}

pub struct SepStats {
    pub total_moves: usize,
    pub total_evals: usize,
}

impl Sum for SepStats {
    fn sum<I: Iterator<Item=SepStats>>(iter: I) -> Self {
        let mut total_moves = 0;
        let mut total_evals = 0;

        for report in iter {
            total_moves += report.total_moves;
            total_evals += report.total_evals;
        }

        SepStats { total_moves, total_evals }
    }
}

impl AddAssign for SepStats {
    fn add_assign(&mut self, other: Self) {
        self.total_moves += other.total_moves;
        self.total_evals += other.total_evals;
    }
}
