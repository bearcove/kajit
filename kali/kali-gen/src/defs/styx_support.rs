use std::hash::{Hash, Hasher};

use facet::Facet;
use facet_reflect::Span;

/// A metadata container that captures both span and doc metadata.
///
/// This is useful for validation errors that need to point back to source locations,
/// while also preserving doc comments.
#[derive(Debug, Clone, Facet)]
#[facet(metadata_container)]
pub struct WithMeta<T> {
    pub value: T,

    #[facet(metadata = "span")]
    pub span: Option<Span>,

    #[facet(metadata = "doc")]
    pub doc: Option<Vec<String>>,

    #[facet(metadata = "tag")]
    pub tag: Option<String>,
}

impl<T: PartialEq> PartialEq for WithMeta<T> {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl<T: Eq> Eq for WithMeta<T> {}

impl<T: Hash> Hash for WithMeta<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.value.hash(state);
    }
}
