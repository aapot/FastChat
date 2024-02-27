import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import argparse

CATEGORIES = ["Writing", "Roleplay", "Reasoning", "Math", "Coding", "Extraction", "STEM", "Humanities"]



def get_model_df(judgment_file):
    cnt = 0
    q2result = []
    fin = open(judgment_file, "r")
    for line in fin:
        obj = json.loads(line)
        obj["category"] = CATEGORIES[(obj["question_id"]-81)//10]
        q2result.append(obj)
    df = pd.DataFrame(q2result)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--judgment-file", type=str, default="data/mt_bench/model_judgment/gpt-4_single.jsonl")
    args = parser.parse_args()

    df = get_model_df(args.judgment_file)
    all_models = df["model"].unique()
    print(all_models)
    scores_all = []
    for model in all_models:
        for cat in CATEGORIES:
            # filter category/model, and score format error (<1% case)
            res = df[(df["category"]==cat) & (df["model"]==model) & (df["score"] >= 0)]
            score = res["score"].mean()

            # # pairwise result
            # res_pair = df_pair[(df_pair["category"]==cat) & (df_pair["model"]==model)]["result"].value_counts()
            # wincnt = res_pair["win"] if "win" in res_pair.index else 0
            # tiecnt = res_pair["tie"] if "tie" in res_pair.index else 0
            # winrate = wincnt/res_pair.sum()
            # winrate_adjusted = (wincnt + tiecnt)/res_pair.sum()
            # # print(winrate_adjusted)

            # scores_all.append({"model": model, "category": cat, "score": score, "winrate": winrate, "wtrate": winrate_adjusted})
            scores_all.append({"model": model, "category": cat, "score": score})

    target_models = [
        # "zephyr-7b-dpo-qlora",
        "OpenHermes-2.5-Mistral-7B",
        # "poro-100pct-dpo-ultrafeedback",
        "poro-100pct-dpo-ultrafeedback-capybara-step1800",
        # "poro-100pct-dpo-dolly-oasst1-oasst2-lima-glaive_eng-fin-en",
        # "poro-100pct-capybara",
        # "poro-100pct-ultrachat",
        "poro-100pct-dolly-oasst2-lima-glaive-step2500",
        # "poro-100pct-dolly-oasst1-oasst2-lima-glaive_eng-fin-step2400-en",
        # "poro-100pct-dolly-oasst2-lima",
        # "llama-7b-finnish-instruct-v3", # start Finnish models
        # "poro-100pct-dolly-oasst1-lima-fi",
        # "poro-100pct-dolly-oasst1-oasst2-lima-glaive_eng-fin-step2400-fi",
        # "poro-100pct-dpo-dolly-oasst1-oasst2-lima-glaive_eng-fin-fi"
    ]

    scores_target = [scores_all[i] for i in range(len(scores_all)) if scores_all[i]["model"] in target_models]

    # sort by target_models
    scores_target = sorted(scores_target, key=lambda x: target_models.index(x["model"]), reverse=True)

    df_score = pd.DataFrame(scores_target)
    df_score = df_score[df_score["model"].isin(target_models)]

    rename_map = {
                # "zephyr-7b-dpo-qlora": "Zephyr-7b-qlora",
                "OpenHermes-2.5-Mistral-7B": "OpenHermes-2.5-Mistral-7B",
                # "poro-100pct-dpo-ultrafeedback": "Poro-DPO-Ultrachat-Ultrafeedback",
                "poro-100pct-dpo-ultrafeedback-capybara-step1800": "Poro-DPO-Capybara-Ultrafeedback",
                # "poro-100pct-dpo-dolly-oasst1-oasst2-lima-glaive_eng-fin-en": "Poro-DPO-Dolly-OASST-LIMA-Glaive_eng-fin",
                # "poro-100pct-capybara": "Poro-SFT-Capybara",
                # "poro-100pct-ultrachat": "Poro-SFT-Ultrachat",
                # "poro-100pct-dolly-oasst2-lima": "Poro-SFT-Dolly-OASST2-LIMA",
                "poro-100pct-dolly-oasst2-lima-glaive-step2500": "Poro-SFT-Dolly-OASST2-LIMA-Glaive",
                # "poro-100pct-dolly-oasst1-oasst2-lima-glaive_eng-fin-step2400-en": "Poro-SFT-Dolly-OASST-LIMA-Glaive_eng-fin",
    }

    # rename_map = {
    #             "llama-7b-finnish-instruct-v3": "Llama-7b-Finnish-Instruct-v0.2",
    #             "poro-100pct-dolly-oasst1-lima-fi": "Poro-SFT-Dolly-OASST1-LIMA-fi",
    #             "poro-100pct-dolly-oasst1-oasst2-lima-glaive_eng-fin-step2400-fi": "Poro-SFT-Dolly-OASST-LIMA-Glaive_eng-fin",
    #             "poro-100pct-dpo-dolly-oasst1-oasst2-lima-glaive_eng-fin-fi": "Poro-DPO-Dolly-OASST-LIMA-Glaive_eng-fin",
    # }

    for k, v in rename_map.items():
        df_score.replace(k, v, inplace=True)

    # plotting results
    fig = px.line_polar(df_score, r = 'score', theta = 'category', line_close = True, category_orders = {"category": CATEGORIES},
                        color = 'model', markers=True, color_discrete_sequence=px.colors.qualitative.Pastel)

    fig.update_layout(
        font=dict(
            size=18,
        ),
    )

    fig.write_image("fig.png", width=900, height=600, scale=1)
    fig.write_html("fig.html")