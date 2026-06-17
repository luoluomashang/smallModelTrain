from small_model_train.chapter_splitter import split_chapters
from small_model_train.dataset_split import split_rows
from small_model_train.preference_builder import build_preference_candidates
from small_model_train.reporting import build_markdown_report
from small_model_train.scoring import score_output
from small_model_train.sft_builder import build_sft_rows


def test_stage_one_pipeline_smoke():
    raw = "第1章 开始\n\n" + "林默说加钱。" * 260
    chapters = split_chapters(raw, work_id="work")
    split = split_rows(chapters, eval_count=1, seed=1)
    card = {
        "id": split[0]["id"],
        "style_contract": "只输出正文。",
        "previous_summary": "上一章结束。",
        "chapter_goal": "完成交易。",
        "target_word_count": "2000-2500中文汉字",
        "chapter_structure": [],
        "character_states": [],
        "must_include": ["加钱"],
        "must_not_include": ["真相"],
        "ending_hook": "箱子响了一下。",
    }
    sft_rows = build_sft_rows([card], split)
    assert len(sft_rows) == 1
    assert sft_rows[0]["output"] == split[0]["text"]

    score = score_output(split[0]["id"], card, split[0]["text"])
    candidates = build_preference_candidates(
        [card],
        [{"id": split[0]["id"], "output": split[0]["text"]}],
        [score],
    )
    report = build_markdown_report("Smoke", [score], {"model": "qwen3"})
    assert "Smoke" in report
    assert "配置快照" in report
    assert isinstance(candidates, list)
