use crate::config::*;
use crate::consts::LBF_SAMPLE_CONFIG;
use crate::optimizer::compress::compression_phase;
use crate::optimizer::explore::exploration_phase;
use crate::optimizer::lbf::LBFBuilder;
use crate::optimizer::separator::Separator;
use crate::util::listener::{OptimizationPhase, ReportType, SolutionListener};
use crate::util::terminator::Terminator;
use jagua_rs::geometry::DTransformation;
use jagua_rs::probs::spp::entities::{SPInstance, SPSolution};
use log::info;
use rand::{Rng, SeedableRng};
use std::time::Duration;
use rand::rngs::Xoshiro256PlusPlus;

pub mod lbf;
pub mod separator;
mod worker;
pub mod explore;
pub mod compress;

///Algorithm 11 from https://doi.org/10.48550/arXiv.2509.13329
///
/// `frozen` pins the given items at fixed transforms for the whole run: they
/// are placed as ordinary items (so real items collide with and pack around
/// them) but are never moved, swapped or shifted by the separators. Item ids
/// `>= min(frozen ids)` are treated as frozen. Pass `&[]` for a normal run.
#[allow(clippy::too_many_arguments)]
pub fn optimize(
    instance: SPInstance,
    mut rng: Xoshiro256PlusPlus,
    sol_listener: &mut impl SolutionListener,
    terminator: &mut impl Terminator,
    expl_config: &ExplorationConfig,
    cmpr_config: &CompressionConfig,
    initial_solution: Option<&SPSolution>,
    frozen: &[(usize, DTransformation)],
) -> SPSolution {
    let mut next_rng = || Xoshiro256PlusPlus::seed_from_u64(rng.next_u64());

    // Any item id at or above the lowest frozen id is a frozen item. Callers
    // append the frozen items after the real ones, so this cleanly partitions
    // the id space; empty frozen list => threshold usize::MAX => nothing pinned.
    let frozen_threshold = frozen.iter().map(|(id, _)| *id).min().unwrap_or(usize::MAX);

    // First build an initial solution if none is provided
    let start_prob = match initial_solution {
        None => {
            let builder = LBFBuilder::new(instance.clone(), next_rng(), LBF_SAMPLE_CONFIG, frozen.to_vec(), frozen_threshold).construct();
            builder.prob
        }
        Some(init_sol) => {
            info!("[OPT] warm starting from provided initial solution");
            let mut prob = jagua_rs::probs::spp::entities::SPProblem::new(instance.clone());
            prob.restore(init_sol);
            prob
        }
    };

    // Begin by executing the exploration phase
    sol_listener.report_phase(OptimizationPhase::Exploration);
    terminator.new_timeout(expl_config.time_limit);
    let mut expl_separator = Separator::new(instance.clone(), start_prob, next_rng(), expl_config.separator_config, frozen_threshold);
    let solutions = exploration_phase(
        &instance,
        &mut expl_separator,
        sol_listener,
        terminator,
        expl_config,
    );
    let final_explore_sol = solutions.last().unwrap().clone();

    // Start the compression phase from the final solution from the exploration phase
    sol_listener.report_phase(OptimizationPhase::Compression);
    terminator.new_timeout(cmpr_config.time_limit);
    let mut cmpr_separator = Separator::new(expl_separator.instance, expl_separator.prob, next_rng(), cmpr_config.separator_config, frozen_threshold);
    let cmpr_sol = compression_phase(
        &instance,
        &mut cmpr_separator,
        &final_explore_sol,
        sol_listener,
        terminator,
        cmpr_config,
    );

    sol_listener.report(ReportType::Final, &cmpr_sol, &instance);

    // Return the final compressed solution
    cmpr_sol
}