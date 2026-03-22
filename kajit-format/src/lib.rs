//! Shared format types and shape utilities for kajit.
//!
//! This crate provides the data types used across kajit frontends (postcard, JSON)
//! and the shape collection utilities that extract field/variant information from
//! facet `Shape` descriptors.

mod shape_utils;
mod types;

pub use shape_utils::*;
pub use types::*;
