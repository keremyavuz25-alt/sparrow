use crate::quantify::overlap_proxy::overlap_area_proxy;
use crate::quantify::simd::circles_soa::CirclesSoA;
use float_cmp::approx_eq;
use jagua_rs::geometry::fail_fast::SPSurrogate;
use jagua_rs::geometry::geo_traits::DistanceTo;
use jagua_rs::geometry::primitives::{Circle, Point};
use std::f32::consts::PI;
use wide::f32x8;

/// Width of the SIMD vector (8 lanes: one AVX register, two NEON registers).
const SIMD_WIDTH: usize = 8;

/// SIMD version of [`overlap_area_proxy`] on the STABLE toolchain (via the `wide` crate).
/// `p2` should match the poles of `sp2`.
#[inline(always)]
pub fn poles_overlap_area_proxy_simd(sp1: &SPSurrogate, sp2: &SPSurrogate, epsilon: f32, p2: &CirclesSoA) -> f32 {
    poles_overlap_area_proxy_simd_bounded(sp1, sp2, epsilon, p2, f32::INFINITY).unwrap()
}

/// Bounded variant: returns `None` as soon as the accumulated (unscaled) overlap
/// exceeds `max_unscaled_overlap` — the caller already knows the sample cannot win.
/// Math is identical to the scalar proxy; only the lane summation order differs.
#[inline(always)]
pub fn poles_overlap_area_proxy_simd_bounded(
    sp1: &SPSurrogate,
    sp2: &SPSurrogate,
    epsilon: f32,
    p2: &CirclesSoA,
    max_unscaled_overlap: f32,
) -> Option<f32> {
    let e_n = f32x8::splat(epsilon);
    let e_sq_n = f32x8::splat(epsilon * epsilon);
    let two_e_n = f32x8::splat(2.0 * epsilon);

    let chunks = p2.x.len() / SIMD_WIDTH;
    let remaining_idx = chunks * SIMD_WIDTH;

    let mut total_overlap = 0.0;
    for p1 in sp1.poles.iter() {
        //common values for all chunks
        let x1_n = f32x8::splat(p1.center.x());
        let y1_n = f32x8::splat(p1.center.y());
        let r1_n = f32x8::splat(p1.radius);

        //process complete chunks with SIMD
        for chunk in 0..chunks {
            let idx = chunk * SIMD_WIDTH;

            // load the next N elements from p2
            let x2 = f32x8::from(<[f32; SIMD_WIDTH]>::try_from(&p2.x[idx..idx + SIMD_WIDTH]).unwrap());
            let y2 = f32x8::from(<[f32; SIMD_WIDTH]>::try_from(&p2.y[idx..idx + SIMD_WIDTH]).unwrap());
            let r2 = f32x8::from(<[f32; SIMD_WIDTH]>::try_from(&p2.r[idx..idx + SIMD_WIDTH]).unwrap());

            // penetration depth
            let dx = x1_n - x2;
            let dy = y1_n - y2;
            let pd = r1_n + r2 - (dx * dx + dy * dy).sqrt();

            // decaying penetration depth: pd if pd >= epsilon, else eps^2 / (2 eps - pd)
            let pd_mask = pd.simd_ge(e_n);
            let decay_values = e_sq_n / (two_e_n - pd);
            let pd_decay = pd_mask.blend(pd, decay_values);

            // weight by the smaller pole
            let min_r = r1_n.min(r2);

            total_overlap += (pd_decay * min_r).reduce_add();
        }

        //process remaining elements with scalar operations
        for j in remaining_idx..p2.x.len() {
            let p2 = Circle {
                center: Point(p2.x[j], p2.y[j]),
                radius: p2.r[j],
            };

            //penetration depth between the two poles (circles)
            let pd = (p1.radius + p2.radius) - p1.center.distance_to(&p2.center);

            let pd_decay = match pd >= epsilon {
                true => pd,
                false => epsilon.powi(2) / (-pd + 2.0 * epsilon),
            };

            total_overlap += pd_decay * f32::min(p1.radius, p2.radius);
        }

        if total_overlap > max_unscaled_overlap {
            return None;
        }
    }

    total_overlap *= PI;

    debug_assert!(
        approx_eq!(f32, total_overlap, overlap_area_proxy(sp1, sp2, epsilon),
                 epsilon = total_overlap * 1e-3),
                  "SIMD and SEQ results do not match: {} vs {}", total_overlap,
                  overlap_area_proxy(sp1, sp2, epsilon)
    );

    debug_assert!(total_overlap.is_normal());
    Some(total_overlap)
}
