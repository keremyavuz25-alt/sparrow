use crate::eval::collision_loss::CollisionLossEvaluator;
use crate::eval::sample_eval::{SampleEval, SampleEvaluator};
use crate::quantify::tracker::CollisionTracker;
use jagua_rs::collision_detection::hazards::collector::BasicHazardCollector;
use jagua_rs::collision_detection::hazards::filter::NoFilter;
use jagua_rs::collision_detection::hazards::{BasicHazardEntity, HazKey};
use jagua_rs::entities::{Item, Layout, PItemKey};
use jagua_rs::geometry::geo_traits::TransformableFrom;
use jagua_rs::geometry::primitives::{Circle, SPolygon};
use jagua_rs::geometry::DTransformation;

pub struct SeparationEvaluator<'a> {
    layout: &'a Layout,
    item: &'a Item,
    collector: BasicHazardCollector,
    current_hazard: (HazKey, BasicHazardEntity),
    loss_evaluator: CollisionLossEvaluator<'a>,
    shape_buff: SPolygon,
    n_evals: usize,
    clear_x: f32,
    clear_y: f32,
    clear_contact: f32,
    current_pk: PItemKey,
}

/// Fork experiment: gap cap (mm) and pole prefix length for the contact proxy.
const CONTACT_CAP: f32 = 100.0;
const CONTACT_POLES: usize = 16;

fn pole_gap(a: &Circle, b: &Circle) -> f32 {
    let dx = a.center.0 - b.center.0;
    let dy = a.center.1 - b.center.1;
    (dx * dx + dy * dy).sqrt() - a.radius - b.radius
}

/// Mean gap (mm, capped) from the sample's surrogate poles to the nearest
/// neighbouring item pole or strip edge (bottom/top/left). Lower = more nestled.
fn contact_score(layout: &Layout, current_pk: PItemKey, shape: &SPolygon) -> f32 {
    let bb = shape.bbox();
    let cont = layout.container().outer_cd().bbox();
    let poles = shape.surrogate().poles.as_slice();
    let mine = &poles[..poles.len().min(CONTACT_POLES)];
    if mine.is_empty() {
        return 0.0;
    }
    let nbs: Vec<&SPolygon> = layout
        .placed_items()
        .iter()
        .filter(|(pk, _)| *pk != current_pk)
        .filter(|(_, pi)| {
            let b = pi.shape().bbox();
            !(b.x_min > bb.x_max + CONTACT_CAP || b.x_max < bb.x_min - CONTACT_CAP || b.y_min > bb.y_max + CONTACT_CAP || b.y_max < bb.y_min - CONTACT_CAP)
        })
        .map(|(_, pi)| pi.shape().as_ref())
        .collect();
    let mut total = 0.0;
    for p in mine {
        let mut g = CONTACT_CAP
            .min(p.center.1 - p.radius - cont.y_min)
            .min(cont.y_max - p.center.1 - p.radius)
            .min(p.center.0 - p.radius - cont.x_min);
        for nb in &nbs {
            for q in nb.surrogate().poles.iter().take(CONTACT_POLES) {
                let d = pole_gap(p, q);
                if d < g {
                    g = d;
                }
            }
        }
        total += g.max(0.0);
    }
    total / mine.len() as f32
}

impl<'a> SeparationEvaluator<'a> {
    pub fn new(
        layout: &'a Layout,
        item: &'a Item,
        current_pk: PItemKey,
        ct: &'a CollisionTracker,
        clear_bias: (f32, f32, f32),
    ) -> Self {
        let current_entity = BasicHazardEntity::from((current_pk, &layout.placed_items()[current_pk]));
        let current_haz_key = layout
            .cde()
            .haz_key(&current_entity)
            .expect("placed item should be registered in the CDE");
        let current_hazard = (current_haz_key, current_entity);

        Self {
            layout,
            item,
            collector: BasicHazardCollector::with_capacity(layout.placed_items().len() + 1),
            current_hazard,
            loss_evaluator: CollisionLossEvaluator::new(layout, ct, current_pk),
            shape_buff: item.shape_cd().as_ref().clone(),
            n_evals: 0,
            clear_x: clear_bias.0,
            clear_y: clear_bias.1,
            clear_contact: clear_bias.2,
            current_pk,
        }
    }
}

impl<'a> SampleEvaluator for SeparationEvaluator<'a> {
    /// Evaluates a transformation. An upper bound can be provided to early terminate the process.
    /// Algorithm 7 from https://doi.org/10.48550/arXiv.2509.13329
    fn evaluate_sample(&mut self, dt: DTransformation, upper_bound: Option<SampleEval>) -> SampleEval {
        self.n_evals += 1;
        let cde = self.layout.cde();

        // Calculate an upper bound of quantification, above which samples are guaranteed to be rejected (because they are dominated by previous ones).
        let loss_bound = match upper_bound {
            Some(SampleEval::Collision { loss }) => loss,
            Some(SampleEval::Clear { .. }) => 0.0,
            _ => f32::INFINITY,
        };
        let shape = self
            .shape_buff
            .transform_from(self.item.shape_cd().as_ref(), &dt.compose());
        self.collector.clear();
        // Mark the moving item's existing hazard as already collected so traversal skips it.
        self.collector.insert(self.current_hazard.0, self.current_hazard.1);
        let n_ignored_hazards = self.collector.len();
        self.loss_evaluator.reload(loss_bound, shape);

        let mut should_stop = |hazard| self.loss_evaluator.add(hazard, shape);
        let stopped_during_surrogate_check = cde.collect_surrogate_collisions_until(
            shape,
            &NoFilter,
            &mut self.collector,
            &mut should_stop,
        );

        match stopped_during_surrogate_check {
            true => SampleEval::Invalid,
            false => {
                let stopped_during_precise_check = cde.collect_poly_collisions_until(
                    shape,
                    &NoFilter,
                    &mut self.collector,
                    &mut should_stop,
                );

                match stopped_during_precise_check {
                    true => SampleEval::Invalid,
                    false if self.collector.len() == n_ignored_hazards => {
                        // Fork experiment: gravity / contact tie-breakers among clear samples.
                        let bb = shape.bbox();
                        let mut loss = self.clear_x * bb.x_max + self.clear_y * bb.y_min;
                        if self.clear_contact > 0.0 {
                            loss += self.clear_contact * contact_score(self.layout, self.current_pk, shape);
                        }
                        SampleEval::Clear { loss }
                    }
                    false => SampleEval::Collision {
                        loss: self.loss_evaluator.loss(),
                    },
                }
            }
        }
    }

    fn n_evals(&self) -> usize {
        self.n_evals
    }
}
