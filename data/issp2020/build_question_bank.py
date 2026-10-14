"""Build the ISSP 2020 question bank with the shared builder (data/issp_common.py): one row per item v1-v60 with the
source-questionnaire wording, answer codes and options from the .dta value labels, and a decision-model question type.
Output: data/issp2020/question_bank.csv

Wording is transcribed from the ISSP 2020 Final Source Questionnaire (ZA7650_bq.pdf, May 2019), which every country
translated from and which is laid out for self-completion. Form instructions ("PLEASE TICK ONE BOX ONLY") are left out;
the answer options are given separately. Placeholders: {country}, {nationality} (e.g. "Australian"), {parliament}
(the national legislature, e.g. "U.S. Congress").
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from issp_common import write_question_bank  # noqa: E402

HERE = Path(__file__).parent                     # data/issp2020/
DTA = HERE / "raw" / "ZA7650_v2-0-0.dta"

AGREE = "How much do you agree or disagree with each of these statements?"
TRUST = ("On a scale of 0 to 10, how much do you personally trust each of the following institutions? 0 means you do not "
         "trust an institution at all, and 10 means you trust it completely.")
CLIMATE = "On a scale from 0 to 10, how bad or good do you think the impacts of climate change will be for {}? 0 means extremely bad, 10 means extremely good."
DANGER = "do you think that {} …"
NEIGHBOURHOOD = ("Thinking about your neighbourhood, to what extent, if at all, was it affected by the following things over "
                 "the last twelve months?")

# v -> (source question number, wording)
SOURCE = {
    "v1": ("Q1a", "Which of these issues is the most important for {country} today?"),
    "v2": ("Q1b", "Which is the next most important?"),
    "v3": ("Q2a", f"{AGREE} Private enterprise is the best way to solve {{country}}'s economic problems."),
    "v4": ("Q2b", f"{AGREE} It is the responsibility of the government to reduce the differences in income between people with high incomes and those with low incomes."),
    "v5": ("Q2c", f"{AGREE} {{country}} should limit the import of foreign products in order to protect its national economy."),
    "v6": ("Q2d", f"{AGREE} {{country}} should limit immigration in order to protect our national way of life."),
    "v7": ("Q2e", f"{AGREE} International organizations are taking away too much power from the {{nationality}} government."),
    "v8": ("Q3a", "Looking at the list below, please tick a box next to the one thing you think should be {country}'s highest priority, the most important thing it should do. {country} should …"),
    "v9": ("Q3b", "And which one do you think should be {country}'s next highest priority, the second most important thing it should do? {country} should …"),
    "v10": ("Q4", "Generally speaking, would you say that most people can be trusted, or that you can't be too careful in dealing with people? Please tick one box to show what you think, where 1 means you can't be too careful and 5 means most people can be trusted."),
    "v11": ("Q5a", f"{TRUST} University research centres"),
    "v12": ("Q5b", f"{TRUST} The news media"),
    "v13": ("Q5c", f"{TRUST} Business and industry"),
    "v14": ("Q5d", f"{TRUST} The {{parliament}}"),
    "v15": ("Q6", "Generally speaking, how concerned are you about environmental issues? Please tick one box below to indicate what you think, where 1 means you are not at all concerned and 5 means you are very concerned."),
    "v16": ("Q7", "Here is a list of some different environmental problems. Which problem, if any, do you think is the most important for {country} as a whole?"),
    "v17": ("Q8", "There has been a lot of discussion about the world's climate and the idea it has been changing in recent decades. Which of the following statements comes closest to your opinion?"),
    "v18": ("Q9a", CLIMATE.format("the world as a whole")),
    "v19": ("Q9b", CLIMATE.format("{country}")),
    "v20": ("Q10a", f"{AGREE} Modern science will solve our environmental problems with little change to our way of life."),
    "v21": ("Q10b", f"{AGREE} We worry too much about the future of the environment and not enough about prices and jobs today."),
    "v22": ("Q10c", f"{AGREE} Almost everything we do in modern life harms the environment."),
    "v23": ("Q10d", f"{AGREE} People worry too much about human progress harming the environment."),
    "v24": ("Q10e", f"{AGREE} In order to protect the environment {{country}} needs economic growth."),
    "v25": ("Q10f", f"{AGREE} Economic growth always harms the environment."),
    "v26": ("Q11a", "How willing would you be to pay much higher prices in order to protect the environment?"),
    "v27": ("Q11b", "And how willing would you be to pay much higher taxes in order to protect the environment?"),
    "v28": ("Q11c", "And how willing would you be to accept cuts in your standard of living in order to protect the environment?"),
    "v29": ("Q11d", "How willing would you be to accept a reduction in the size of {country}'s protected nature areas, in order to open them up for economic development?"),
    "v30": ("Q12a", f"{AGREE} It is just too difficult for someone like me to do much about the environment."),
    "v31": ("Q12b", f"{AGREE} I do what is right for the environment, even when it costs more money or takes more time."),
    "v32": ("Q12c", f"{AGREE} There are more important things to do in life than protect the environment."),
    "v33": ("Q12d", f"{AGREE} There is no point in doing what I can for the environment unless others do the same."),
    "v34": ("Q12e", f"{AGREE} Many of the claims about environmental threats are exaggerated."),
    "v35": ("Q12f", f"{AGREE} I find it hard to know whether the way I live is helpful or harmful to the environment."),
    "v36": ("Q12g", f"{AGREE} Environmental problems have a direct effect on my everyday life."),
    "v37": ("Q13a", "In general, " + DANGER.format("air pollution caused by cars is")),
    "v38": ("Q13b", "In general, " + DANGER.format("air pollution caused by industry is")),
    "v39": ("Q13c", "And " + DANGER.format("pesticides and chemicals used in farming are")),
    "v40": ("Q13d", "And " + DANGER.format("pollution of {country}'s rivers, lakes and streams is")),
    "v41": ("Q13e", "In general, " + DANGER.format("a rise in the world's temperature caused by climate change is")),
    "v42": ("Q13f", "And " + DANGER.format("modifying the genes of certain crops is")),
    "v43": ("Q13g", "And " + DANGER.format("nuclear power stations are")),
    "v44": ("Q14a", "Which of these approaches do you think would be the best way of getting business and industry in {country} to protect the environment?"),
    "v45": ("Q14b", "Which of these approaches do you think would be the best way of getting people and their families in {country} to protect the environment?"),
    "v46": ("Q15", "How much, if at all, do you enjoy being outside in nature?"),
    "v47": ("Q16", "In the last twelve months how often, if at all, have you engaged in any leisure activities outside in nature, such as hiking, bird watching, swimming, skiing, other outdoor activities or just relaxing?"),
    "v48": ("Q17a", "In the last twelve months, how many trips did you make by plane? Count outward and return journeys, including transfers, as one trip."),
    "v49": ("Q17b", "In a typical week, about how many hours do you spend in a car or another motor vehicle, including motorcycles, trucks, and vans, but not counting public transport? Do not include shared rides in buses, minibuses, and collective taxis."),
    "v50": ("Q17c", "In a typical week, on how many days do you eat beef, lamb, or products that contain them?"),
    "v51": ("Q18", "How many rooms are there in your home (apartment or house)? Do not count any separate kitchens, bathrooms, garages, balconies, hallways or cupboards."),
    "v52": ("Q19a", "How often do you make a special effort to sort glass or tins or plastic or newspapers and so on for recycling?"),
    "v53": ("Q19b", "And how often do you avoid buying certain products for environmental reasons?"),
    "v54": ("Q20", "Are you a member of any group whose main aim is to preserve or protect the environment?"),
    "v55": ("Q21a", "In the last five years, have you signed a petition about an environmental issue?"),
    "v56": ("Q21b", "In the last five years, have you given money to an environmental group?"),
    "v57": ("Q21c", "In the last five years, have you taken part in a protest or demonstration about an environmental issue?"),
    "v58": ("Q22a", f"{NEIGHBOURHOOD} Air pollution"),
    "v59": ("Q22b", f"{NEIGHBOURHOOD} Water pollution"),
    "v60": ("Q22c", f"{NEIGHBOURHOOD} Extreme weather events (such as severe storms, droughts, floods, heat waves, cold snaps, etc.)"),
}

CHOICE = {"v1", "v8", "v16", "v17", "v44", "v45"}
NOUL = {"v54", "v55", "v56", "v57"}
EXCLUDE = {"v2": "depends on answer to v1", "v9": "depends on answer to v8",
           "v48": "open count (plane trips)", "v49": "open count (car hours)", "v51": "open count (rooms)"}
# first label of a score scale that starts at the high end -> reverse so levels run low to high
HIGH_FIRST = ("Agree strongly", "Very willing", "Extremely dangerous", "Always", "Daily")


if __name__ == "__main__":
    write_question_bank(HERE, DTA, HERE / "raw" / "ZA7650_v2-0-0_missing.txt",
                        {v: question for v, (_, question) in SOURCE.items()}, CHOICE, NOUL, EXCLUDE, HIGH_FIRST)
