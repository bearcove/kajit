use std::fs;
use std::hash::{Hash, Hasher};
use std::path::Path;

use facet::Facet;
use facet_reflect::Span;
use facet_styx::{Documented, RenderError};
use indexmap::IndexMap;

/// A definitionlanguage like HIR, IR, MIR
#[derive(Facet, Debug, Clone)]
pub struct LangSpec {
    /// Name of the langauge
    pub name: String,

    /// File extension, excluding the dot: for `sample.k-hir` it's just `k-hir`
    pub file_ext: String,

    /// Short description of the lang
    pub description: String,

    /// Grammar rules (and optionally embedded templates)
    pub rules: ModernRulesDecl,
}

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

#[derive(Facet, Debug, Clone)]
pub struct ModernRulesDecl {
    pub templates: Option<IndexMap<Documented<String>, TemplateDecl>>,

    #[facet(flatten)]
    pub rules: IndexMap<Documented<String>, RuleDecl>,
}

#[derive(Facet, Debug, Clone)]
pub struct TemplateDecl {
    pub syntax: SyntaxExpr,
    pub highlight: Option<String>,
}

#[derive(Facet, Debug, Clone)]
pub struct TemplateSyntaxDecl {
    pub params: Vec<SyntaxExpr>,
    pub body: HighlightedSyntax,
}

#[derive(Facet, Debug, Clone)]
pub struct HighlightedSyntax {
    pub syntax: Option<SyntaxExpr>,
    pub highlight: Option<String>,
}

#[derive(Facet, Debug, Clone)]
#[facet(rename_all = "lowercase")]
#[repr(u8)]
pub enum RuleDecl {
    #[facet(untagged)]
    Literal(String),
    #[facet(untagged)]
    InlineStruct(HighlightedSyntax),
    Struct(HighlightedSyntax),
    Enum,
    #[facet(other)]
    UserTag {
        #[facet(tag)]
        tag: String,
        #[facet(content)]
        content: Option<SyntaxPayload>,
    },
}

#[derive(Facet, Debug, Clone)]
#[facet(rename_all = "snake_case")]
#[repr(u8)]
pub enum SyntaxExpr {
    Template(Box<TemplateSyntaxDecl>),
    Regex(Vec<SyntaxExpr>),
    BraceBlock(Vec<SyntaxExpr>),
    ParenList(Vec<SyntaxExpr>),
    BracketList(Vec<SyntaxExpr>),
    Paren(Vec<SyntaxExpr>),
    Bracket(Vec<SyntaxExpr>),
    Optional(Vec<SyntaxExpr>),
    Many(Vec<SyntaxExpr>),
    SeparatedBy(Vec<SyntaxExpr>),
    Indent(Vec<SyntaxExpr>),
    SoftSpace,
    Newline,
    #[facet(rename = "Any")]
    Any,
    #[facet(rename = "Rule")]
    Rule,
    #[facet(other)]
    Other {
        #[facet(tag)]
        tag: Option<String>,
        #[facet(content)]
        content: Option<SyntaxPayload>,
    },
}

#[derive(Facet, Debug, Clone)]
pub struct SyntaxObject {
    #[facet(flatten)]
    pub fields: IndexMap<Documented<String>, SyntaxExpr>,
}

#[derive(Facet, Debug, Clone)]
#[repr(u8)]
#[facet(untagged)]
pub enum SyntaxPayload {
    Scalar(String),
    Seq(Vec<SyntaxExpr>),
    Object(SyntaxObject),
}

pub fn read_from_file(schema_path: &Path) -> Result<LangSpec, String> {
    let schema_source = fs::read_to_string(schema_path)
        .map_err(|e| format!("failed to read {}: {e}", schema_path.display()))?;
    match facet_styx::from_str::<LangSpec>(&schema_source) {
        Ok(schema) => Ok(schema),
        Err(err) => Err(format!(
            "failed to parse {} as Styx:\n{}",
            schema_path.display(),
            err.render(&schema_path.display().to_string(), &schema_source),
        )),
    }
}
