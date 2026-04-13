use std::fs;
use std::path::Path;

use facet::Facet;
use facet_styx::{Documented, RenderError};
use indexmap::IndexMap;

#[derive(Facet, Debug, Clone)]
#[facet(metadata_container)]
pub(crate) struct WithDoc<T> {
    pub(crate) value: T,

    #[facet(metadata = "doc")]
    pub(crate) doc: Option<Vec<String>>,
}

#[derive(Facet, Debug, Clone)]
#[allow(dead_code)]
pub(crate) struct ModernPilotSchemaDocument {
    pub(crate) name: String,
    pub(crate) file_ext: String,
    pub(crate) description: String,
    pub(crate) rules: ModernRulesDecl,
}

#[derive(Facet, Debug, Clone)]
#[allow(dead_code)]
pub(crate) struct ModernRulesDecl {
    pub(crate) templates: Option<IndexMap<Documented<String>, TemplateDecl>>,

    #[facet(flatten)]
    pub(crate) rules: IndexMap<Documented<String>, ModernRuleDecl>,
}

#[derive(Facet, Debug, Clone)]
#[allow(dead_code)]
pub(crate) struct TemplateDecl {
    pub(crate) syntax: SyntaxExpr,
    pub(crate) highlight: Option<String>,
}

#[derive(Facet, Debug, Clone)]
#[allow(dead_code)]
pub(crate) struct TemplateSyntaxDecl {
    pub(crate) params: Vec<SyntaxExpr>,
    pub(crate) body: ModernStructDecl,
}

#[derive(Facet, Debug, Clone)]
#[allow(dead_code)]
pub(crate) struct ModernStructDecl {
    pub(crate) syntax: Option<SyntaxExpr>,
    pub(crate) highlight: Option<String>,
}

#[derive(Facet, Debug, Clone)]
#[allow(dead_code)]
#[facet(rename_all = "lowercase")]
#[repr(u8)]
pub(crate) enum ModernRuleDecl {
    #[facet(untagged)]
    Literal(String),
    #[facet(untagged)]
    InlineStruct(ModernStructDecl),
    Struct(ModernStructDecl),
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
#[allow(dead_code)]
pub(crate) struct Schema {
    pub(crate) name: String,
    pub(crate) file_ext: String,
    pub(crate) description: String,
    pub(crate) templates: IndexMap<Documented<String>, TemplateDecl>,
    pub(crate) rules: IndexMap<Documented<String>, ModernRuleDecl>,
}

#[derive(Facet, Debug, Clone)]
#[allow(dead_code)]
#[facet(rename_all = "snake_case")]
#[repr(u8)]
pub(crate) enum SyntaxExpr {
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
#[allow(dead_code)]
pub(crate) struct SyntaxObject {
    #[facet(flatten)]
    pub(crate) fields: IndexMap<Documented<String>, SyntaxExpr>,
}

#[derive(Facet, Debug, Clone)]
#[allow(dead_code)]
#[repr(u8)]
#[facet(untagged)]
pub(crate) enum SyntaxPayload {
    Scalar(String),
    Seq(Vec<SyntaxExpr>),
    Object(SyntaxObject),
}

pub(crate) fn read_from_file(schema_path: &Path) -> Result<Schema, String> {
    let schema_source = fs::read_to_string(schema_path)
        .map_err(|e| format!("failed to read {}: {e}", schema_path.display()))?;
    match facet_styx::from_str::<ModernPilotSchemaDocument>(&schema_source) {
        Ok(schema) => validate_modern_repr_schema(&schema, schema_path),
        Err(err) => Err(format!(
            "failed to parse {} as Styx:\n{}",
            schema_path.display(),
            err.render(&schema_path.display().to_string(), &schema_source),
        )),
    }
}

fn validate_modern_repr_schema(
    schema: &ModernPilotSchemaDocument,
    path: &Path,
) -> Result<Schema, String> {
    if schema.name.trim().is_empty() {
        return Err(format!("expected {} name to be non-empty", path.display()));
    }

    if !schema.file_ext.starts_with('.') || schema.file_ext.len() < 2 {
        return Err(format!(
            "expected {} file_ext to start with '.', got {:?}",
            path.display(),
            schema.file_ext
        ));
    }

    if schema.description.trim().is_empty() {
        return Err(format!(
            "expected {} description to be non-empty",
            path.display()
        ));
    }

    if schema.rules.rules.is_empty() {
        return Err(format!(
            "expected {} rules to contain at least one rule",
            path.display()
        ));
    }

    Ok(Schema {
        name: schema.name.clone(),
        file_ext: schema.file_ext.clone(),
        description: schema.description.clone(),
        templates: schema.rules.templates.clone().unwrap_or_default(),
        rules: schema.rules.rules.clone(),
    })
}
