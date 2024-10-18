import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from utils import *
from run import *


def load_runs(filename_prefix: str) -> pd.DataFrame:

    total_runs = 0
    file_dne_skips = 0
    persona_collision_skips = 0

    df = pd.DataFrame(columns=["turns", "label", "persona_id", "user_id"])
    for persona_id in tqdm(range(100)):
        persona, probe_str, judge_func = personas[persona_id]
        for user_id in range(100):
            if user_id == persona_id:
                persona_collision_skips += 1
                continue
            # user, probe_str_user, judge_func_user = personas[user_id]
            # P1
            file_name = (
                f"{filename_prefix}_agent_{persona_id}_user_{user_id}_turn_32.pkl"
            )
            if Path(file_name).exists():
                with open(file_name, "rb") as handle:
                    pkl = pickle.load(handle)
            else:
                file_dne_skips += 1
                continue

            total_runs += 1

            for i, answers in pkl["probed_history_per_turn"].items():
                for answer in answers:
                    # TODO: should we actually modulate by fluency?
                    # fluency = float(is_fluent_english(answer))
                    # label = float(judge_func(answer)) * fluency
                    label = float(judge_func(answer))
                    df = pd.concat(
                        [
                            df,
                            pd.DataFrame(
                                {
                                    "turns": [i],
                                    "label": [label],
                                    "persona_id": [persona_id],
                                    "user_id": [user_id],
                                }
                            ),
                        ],
                        ignore_index=True,
                    )

    print(f"Total runs: {total_runs}")
    print(f"Skipped {persona_collision_skips} due to persona collisions.")
    return df


def plot_runs(model_to_df: dict[str, pd.DataFrame]):
    fs = 20
    sns.set_style("darkgrid", {"axes.facecolor": ".95"})
    fig, ax = plt.subplots(figsize=(10, 7.5), dpi=80, facecolor="w", edgecolor="k")
    plt.rcParams["font.size"] = fs

    for model, df in model_to_df.items():
        sns.lineplot(
            data=df, x="turns", y="label", errorbar=("ci", 68), ax=ax, label=model
        )

    ref_df = model_to_df["Mixtral-8x22B-Instruct-v0.1"]
    ax.set_xlabel("Number of Turns", fontsize=fs)
    ax.set_ylabel("Persona Stability", fontsize=fs)
    ax.tick_params(axis="x", labelsize=fs)
    ax.tick_params(axis="y", labelsize=fs)
    ax.set_xlim(right=ref_df["turns"].max(), left=ref_df["turns"].min())

    plt.show()


def main():
    model_to_filename_prefix = {
        "Mixtral-8x22B-Instruct-v0.1": "selfchat/mistralai-Mixtral-8x22B-Instruct-v0.1",
        "gpt-3.5-turbo-16k": "selfchat/gpt-3.5-turbo-16k",
        "Mistral-7B-Instruct-v0.3": "results/selfchat/mistralai-Mistral-7B-Instruct-v0.3",
    }

    model_to_df = {}
    for model, filename_prefix in model_to_filename_prefix.items():
        print(model)
        df = load_runs(filename_prefix)
        unique_pairs = df[["persona_id", "user_id"]].drop_duplicates()
        number_of_unique_pairs = len(unique_pairs)
        print(number_of_unique_pairs)
        model_to_df[model] = df
        print("---")

    plot_runs(model_to_df)


if __name__ == "__main__":
    main()
