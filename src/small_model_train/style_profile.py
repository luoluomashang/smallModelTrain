"""作者风格画像与风格契约生成。

风格画像把正文集合压缩成少量可解释统计；风格契约再把这些统计转成给模型
看的写作约束。第一阶段先做“能稳定跑”的模板，后续可继续增加更细指标。
"""

from __future__ import annotations

from statistics import mean

from small_model_train.text_utils import (
    count_chinese_chars,
    dialogue_ratio,
    paragraph_lengths,
)


def build_style_profile(rows: list[dict]) -> dict:
    """从章节正文中提取平均字数、段落长度和对白比例。"""

    texts = [row.get("text", "") for row in rows if row.get("text")]
    paragraph_counts = [length for text in texts for length in paragraph_lengths(text)]
    return {
        "chapter_count": len(texts),
        "avg_chinese_chars": round(mean([count_chinese_chars(text) for text in texts]), 2)
        if texts
        else 0,
        "avg_paragraph_chars": round(mean(paragraph_counts), 2) if paragraph_counts else 0,
        "avg_dialogue_ratio": round(mean([dialogue_ratio(text) for text in texts]), 4)
        if texts
        else 0,
    }


def render_style_contract(profile: dict) -> str:
    """把风格统计渲染成可直接放进章节卡的中文契约。"""

    dialogue_percent = round(float(profile.get("avg_dialogue_ratio", 0)) * 100, 1)
    avg_paragraph_chars = profile.get("avg_paragraph_chars", 0)
    return "\n".join(
        [
            "【角色】",
            "你是作者的正文执行器，只负责根据章节执行卡写正文。",
            "",
            "【叙述原则】",
            "1. 句子朴素直接，动作承接优先于心理解释。",
            "2. 情绪通过动作、对白和反应表现，不写总结式升华。",
            "3. 主角视角跟随，不随意跳到全知视角。",
            f"4. 段落长度参考：平均约 {avg_paragraph_chars} 个中文汉字。",
            "",
            "【对白原则】",
            f"1. 对话比例参考：约 {dialogue_percent}%。",
            "2. 对话短、准、自然，不用长篇对白解释世界观。",
            "3. 允许省略、打断和反问。",
            "",
            "【禁止风格】",
            "1. 不写空气仿佛凝固了。",
            "2. 不写难以言喻的情绪涌上心头。",
            "3. 不写命运的齿轮开始转动。",
            "4. 不写嘴角勾起一抹弧度。",
            "5. 不写眼神逐渐坚定起来。",
            "",
            "【输出要求】",
            "只输出正文。不要输出提纲、小标题、解释、分析或提示语。",
        ]
    )
