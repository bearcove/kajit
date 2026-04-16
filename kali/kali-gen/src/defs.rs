use std::fs;
use std::hash::{Hash, Hasher};

use camino::Utf8Path;
use facet::Facet;
use facet_reflect::Span;
use facet_styx::{Documented, RenderError};
use indexmap::IndexMap;

use crate::defs::styx_support::WithMeta;

mod styx_support;

/// A language like HIR, IR, MIR
#[derive(Facet, Debug, Clone)]
pub struct LangDef {
    /// Name of the langauge
    pub name: String,

    /// File extension, excluding the dot: for `sample.k-hir` it's just `k-hir`
    pub file_ext: String,

    /// Short description of the lang
    pub description: String,

    /// Grammar rules (and optionally embedded templates)
    pub rules: RulesDef,
}

#[derive(Facet, Debug, Clone)]
pub struct RulesDef {
    #[facet(default)]
    pub templates: IndexMap<WithMeta<String>, WithMeta<TemplateDef>>,

    #[facet(default)]
    pub rules: IndexMap<WithMeta<String>, WithMeta<RuleDef>>,
}

#[derive(Facet, Debug, Clone)]
pub struct TemplateDef {
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
pub enum RuleDef {
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

pub fn read_from_file(schema_path: &Utf8Path) -> Result<LangDef, String> {
    let schema_source = fs::read_to_string(schema_path)
        .map_err(|e| format!("failed to read {}: {e}", schema_path))?;

    match facet_styx::from_str::<LangDef>(&schema_source) {
        Ok(schema) => Ok(schema),
        Err(err) => Err(format!(
            "failed to parse {} as Styx:\n{}",
            schema_path,
            err.render(&schema_path.to_string(), &schema_source),
        )),
    }
}
