use facet::Facet;
use serde::{Deserialize, Serialize};

#[derive(Debug, PartialEq, Serialize, Deserialize, Facet)]
struct Nums {
    vals: Vec<u32>,
}

#[test]
fn postcard_vec_u32_v0_regalloc_differential_runs() {
    let value = Nums {
        vals: vec![1, 2, 3],
    };
    let input = postcard::to_allocvec(&value).expect("serialize postcard input");
    let linear = kajit::debug_linear_ir(Nums::SHAPE, &kajit::postcard::KajitPostcard);
    let cfg_program = kajit::regalloc_engine::cfg_mir::lower_linear_ir(&linear);
    let lambda_ids: Vec<usize> = cfg_program
        .funcs
        .iter()
        .map(|f| f.lambda_id.index())
        .collect();
    assert!(
        cfg_program.funcs.len() > 1,
        "expected lambda body functions in CFG program; ids={lambda_ids:?}"
    );
    assert!(
        lambda_ids.contains(&1),
        "expected lambda id @1 in CFG program; ids={lambda_ids:?}"
    );
    let ra_program = kajit_mir::lower_linear_ir(&linear);
    let report = kajit::regalloc_engine::differential_check_program(ra_program, &input);
    match report {
        kajit::regalloc_engine::DifferentialCheckResult::Match { .. }
        | kajit::regalloc_engine::DifferentialCheckResult::Diverged(_) => {}
        kajit::regalloc_engine::DifferentialCheckResult::Error(err) => {
            panic!("regalloc differential harness should execute: {err}");
        }
    }
}

#[test]
fn postcard_vec_u32_v0_linear_49_has_term_cond_alloc() {
    let value = Nums {
        vals: vec![1, 2, 3],
    };
    let _input = postcard::to_allocvec(&value).expect("serialize postcard input");
    let linear = kajit::debug_linear_ir(Nums::SHAPE, &kajit::postcard::KajitPostcard);
    let cfg_program = kajit::regalloc_engine::cfg_mir::lower_linear_ir(&linear);
    let allocated =
        kajit::regalloc_engine::allocate_cfg_program(&cfg_program).expect("allocate cfg program");
    let cfg_lambda1 = cfg_program
        .funcs
        .iter()
        .find(|f| f.lambda_id.index() == 1)
        .expect("lambda @1 cfg function should exist");
    let schedule = cfg_lambda1
        .derive_schedule()
        .expect("lambda @1 schedule should derive");
    let Some(op_id) = schedule.op_order.get(49).copied() else {
        panic!(
            "lambda @1 should have schedule index 49, len={}",
            schedule.op_order.len()
        );
    };
    let lambda1 = allocated
        .functions
        .iter()
        .find(|f| f.lambda_id.index() == 1)
        .expect("lambda @1 alloc should exist");

    let chosen_len;
    let mut raw = Vec::new();
    let allocs_len = lambda1
        .op_allocs
        .get(&op_id)
        .map(|allocs| allocs.len())
        .unwrap_or(0);
    let is_term = matches!(op_id, kajit::regalloc_engine::cfg_mir::OpId::Term(_));
    raw.push((49usize, is_term, allocs_len));
    chosen_len = allocs_len;
    assert!(
        chosen_len >= 1,
        "lambda @1 linear 49 should expose at least one alloc for branch cond; raw={raw:?}, chosen_len={chosen_len}"
    );
}
