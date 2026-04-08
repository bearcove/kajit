pub mod const_fold;
pub mod dead_theta_ports;
pub mod gamma_output_partition;
pub mod hints;
pub mod interpret;
pub(crate) mod ir_passes;
pub mod post_unroll_canonicalize;
pub mod provenance;
pub mod reduce;
pub mod simplify_gamma;
pub mod unroll_theta;
mod verify;

pub use ir_passes::*;
pub use verify::*;
