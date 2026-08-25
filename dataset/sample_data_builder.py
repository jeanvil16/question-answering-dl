"""Build dataset/sample_squad.json - a small SQuAD-format demo dataset.

The project ships with a hand-written, SQuAD-v1.1-compatible file so the
whole pipeline (training -> backend -> frontend) can run offline on any
laptop in under a minute. Answer character offsets are computed
programmatically with `context.index(answer)`, guaranteeing they are correct.

For real experiments download the full dataset:
    python dataset/download_squad.py

Run:  python dataset/sample_data_builder.py
"""

import json
import re
from pathlib import Path

# Each passage is a short encyclopaedic paragraph; each QA pair's answer MUST
# appear verbatim inside the passage (asserted below).
PASSAGES = [
    {
        "title": "Photosynthesis",
        "context": (
            "Photosynthesis is the process used by plants, algae and some bacteria "
            "to convert light energy into chemical energy. It takes place inside "
            "organelles called chloroplasts, which contain a green pigment called "
            "chlorophyll. During photosynthesis, plants absorb carbon dioxide from "
            "the air and water from the soil, and release oxygen as a by-product. "
            "The glucose produced is used for growth and energy storage. "
            "Photosynthesis is essential for life on Earth because it provides food "
            "and oxygen for nearly every living organism."
        ),
        "qas": [
            ("What process converts light energy into chemical energy?", "photosynthesis"),
            ("Where does photosynthesis take place?", "chloroplasts"),
            ("Which pigment absorbs light in photosynthesis?", "chlorophyll"),
            ("What gas do plants release during photosynthesis?", "oxygen"),
            ("What do plants absorb from the air?", "carbon dioxide"),
            ("Which sugar is produced by photosynthesis?", "glucose"),
            ("Why is photosynthesis essential for life on Earth?",
             "because it provides food and oxygen for nearly every living organism"),
        ],
    },
    {
        "title": "History of the Internet",
        "context": (
            "The Internet began in the late 1960s as ARPANET, a research project "
            "funded by the United States Department of Defense. ARPANET sent its "
            "first message between two universities in 1969. In 1983 the network "
            "adopted TCP/IP, the communication protocol that still powers the modern "
            "Internet. The World Wide Web was invented by Tim Berners-Lee in 1989 "
            "while he was working at CERN. By the mid 1990s the web had opened the "
            "Internet to ordinary users, and today billions of people rely on it for "
            "communication, commerce and entertainment."
        ),
        "qas": [
            ("What was the Internet originally called?", "ARPANET"),
            ("Who funded the ARPANET research project?",
             "the United States Department of Defense"),
            ("In which year did ARPANET send its first message?", "1969"),
            ("Which communication protocol was adopted in 1983?", "TCP/IP"),
            ("Who invented the World Wide Web?", "Tim Berners-Lee"),
            ("When was the World Wide Web invented?", "1989"),
            ("Where did Tim Berners-Lee work when he invented the web?", "CERN"),
        ],
    },
    {
        "title": "Taj Mahal",
        "context": (
            "The Taj Mahal is a white marble mausoleum located in Agra, India, on the "
            "southern bank of the Yamuna River. It was commissioned in 1632 by the "
            "Mughal emperor Shah Jahan in memory of his favourite wife, Mumtaz Mahal, "
            "who died giving birth to their fourteenth child. Construction employed "
            "about twenty thousand artisans and was largely completed in 1653. Widely "
            "admired as the jewel of Muslim art in India, the Taj Mahal was designated "
            "a UNESCO World Heritage Site in 1983 and attracts millions of visitors "
            "every year."
        ),
        "qas": [
            ("Who commissioned the Taj Mahal?", "Shah Jahan"),
            ("In whose memory was the Taj Mahal built?", "Mumtaz Mahal"),
            ("In which year was construction of the Taj Mahal commissioned?", "1632"),
            ("Where is the Taj Mahal located?", "Agra, India"),
            ("On which river is the Taj Mahal situated?", "the Yamuna River"),
            ("How many artisans worked on the construction?",
             "about twenty thousand"),
            ("When did the Taj Mahal become a UNESCO World Heritage Site?", "1983"),
            ("What material is the Taj Mahal made of?", "white marble"),
        ],
    },
    {
        "title": "The Solar System",
        "context": (
            "The solar system consists of the Sun and everything that orbits around "
            "it, including eight planets. Jupiter is the largest planet, while Mercury "
            "is the smallest. Venus is the hottest planet because its thick atmosphere "
            "traps heat, whereas Neptune is the coldest and lies farthest from the Sun. "
            "Earth is the only planet known to support life. Saturn is famous for its "
            "spectacular ring system, and Mars is often called the red planet because "
            "iron minerals in its soil rust, giving the surface a reddish colour."
        ),
        "qas": [
            ("What is at the centre of the solar system?", "the Sun"),
            ("Which planet is the largest?", "Jupiter"),
            ("Which planet is the smallest?", "Mercury"),
            ("Which planet is the hottest?", "Venus"),
            ("Why is Venus the hottest planet?", "its thick atmosphere traps heat"),
            ("Which planet lies farthest from the Sun?", "Neptune"),
            ("Which planet has a spectacular ring system?", "Saturn"),
            ("Which planet is called the red planet?", "Mars"),
            ("Which planet is known to support life?", "Earth"),
        ],
    },
    {
        "title": "Machine Learning",
        "context": (
            "Machine learning is a branch of artificial intelligence in which "
            "computers learn patterns from data instead of being programmed with "
            "explicit rules. In supervised learning a model is trained on labelled "
            "examples, while unsupervised learning finds hidden structure in "
            "unlabelled data. Reinforcement learning trains an agent through rewards "
            "and penalties. Deep learning uses artificial neural networks with many "
            "layers, known as deep neural networks, to learn rich representations. A "
            "model improves during training by adjusting its parameters to minimise a "
            "loss function, and its skill is measured on unseen test data."
        ),
        "qas": [
            ("What branch of artificial intelligence learns patterns from data?",
             "Machine learning"),
            ("What kind of examples are used in supervised learning?",
             "labelled examples"),
            ("What does unsupervised learning find?",
             "hidden structure in unlabelled data"),
            ("How does reinforcement learning train an agent?",
             "through rewards and penalties"),
            ("What does deep learning use to learn rich representations?",
             "artificial neural networks with many layers"),
            ("What does a model minimise during training?", "a loss function"),
            ("Where is a model's skill measured?", "on unseen test data"),
        ],
    },
    {
        "title": "The Human Heart",
        "context": (
            "The human heart is a muscular organ roughly the size of a fist that pumps "
            "blood through the circulatory system. It contains four chambers: two upper "
            "atria and two lower ventricles. The left ventricle has the thickest walls "
            "because it pushes oxygen-rich blood into the aorta to supply the whole "
            "body. Deoxygenated blood returns through veins and is sent to the lungs by "
            "the right side of the heart. A healthy adult heart beats about seventy two "
            "times per minute, roughly one hundred thousand times each day."
        ),
        "qas": [
            ("What pumps blood through the circulatory system?", "The human heart"),
            ("How many chambers does the heart have?", "four chambers"),
            ("What are the upper chambers of the heart called?", "atria"),
            ("Which part of the heart has the thickest walls?", "The left ventricle"),
            ("Into which artery does the left ventricle push blood?", "the aorta"),
            ("Through which vessels does deoxygenated blood return?", "veins"),
            ("How many times per minute does a healthy adult heart beat?",
             "about seventy two times per minute"),
        ],
    },
    {
        "title": "Mount Everest",
        "context": (
            "Mount Everest is the highest mountain above sea level, with a summit "
            "elevation of 8,848 metres. It lies in the Himalayas on the border between "
            "Nepal and Tibet, an autonomous region of China. Known in Nepali as "
            "Sagarmatha and in Tibetan as Chomolungma, Everest was first summited on "
            "29 May 1953 by Edmund Hillary of New Zealand and Tenzing Norgay, a Sherpa "
            "mountaineer from Nepal. The mountain continues to rise a few millimetres "
            "every year because of tectonic activity beneath the Himalayas."
        ),
        "qas": [
            ("Which is the highest mountain above sea level?", "Mount Everest"),
            ("What is the summit elevation of Mount Everest?", "8,848 metres"),
            ("In which mountain range does Everest lie?", "the Himalayas"),
            ("Between which two regions does Everest stand on the border?",
             "Nepal and Tibet"),
            ("When was Mount Everest first summited?", "29 May 1953"),
            ("Who were the first climbers to reach the summit?",
             "Edmund Hillary of New Zealand and Tenzing Norgay"),
            ("What is Mount Everest called in Nepali?", "Sagarmatha"),
            ("Why does Everest keep rising every year?",
             "because of tectonic activity beneath the Himalayas"),
        ],
    },
    {
        "title": "The Pacific Ocean",
        "context": (
            "The Pacific Ocean is the largest and deepest ocean on Earth, covering more "
            "than sixty three million square miles, about one third of the planet's "
            "surface. Its deepest point, the Challenger Deep in the Mariana Trench, "
            "reaches roughly eleven kilometres below sea level, deeper than Mount "
            "Everest is tall. The Pacific is bordered by Asia and Australia on the west "
            "and the Americas on the east. Because it sits atop the Ring of Fire, the "
            "ocean experiences frequent earthquakes and volcanic eruptions, and its "
            "warm surface waters fuel powerful typhoons."
        ),
        "qas": [
            ("Which is the largest ocean on Earth?", "The Pacific Ocean"),
            ("What is the deepest point of the Pacific Ocean?", "the Challenger Deep"),
            ("In which trench is the Challenger Deep located?", "the Mariana Trench"),
            ("How deep is the Challenger Deep?",
             "roughly eleven kilometres below sea level"),
            ("Which continents border the Pacific on the west?", "Asia and Australia"),
            ("Which region borders the Pacific on the east?", "the Americas"),
            ("What is the name of the volcanic belt under the Pacific?",
             "the Ring of Fire"),
            ("What fuels powerful typhoons over the Pacific?",
             "its warm surface waters"),
        ],
    },
]


def slugify(title):
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def paraphrase(question):
    """Simple template-based question augmentation.

    With only ~60 QA pairs the network would otherwise MEMORIZE exact
    question->span mappings instead of learning to match question words
    against context words. Adding cheap paraphrases triples the effective
    dataset and forces the attention layer to rely on question CONTENT.
    """
    q = question.strip()
    lower = q[0].lower() + q[1:] if q else q
    return [
        q,
        f"Based on the passage, {lower}",
        f"Please answer: {q}",
    ]


def build():
    data, qa_counter = [], 0
    for passage in PASSAGES:
        context = passage["context"]
        qas = []
        for question, answer in passage["qas"]:
            answer_start = context.index(answer)   # raises if answer missing
            for variant in paraphrase(question):
                qa_counter += 1
                qas.append({
                    "id": f"{slugify(passage['title'])}-{qa_counter}",
                    "question": variant,
                    "answers": [{"text": answer, "answer_start": answer_start}],
                    "is_impossible": False,
                })
        data.append({
            "title": passage["title"],
            "paragraphs": [{"context": context, "qas": qas}],
        })

    return {"version": "1.1", "data": data}


def main():
    out_path = Path(__file__).parent / "sample_squad.json"
    squad = build()
    n_qas = sum(len(p["qas"]) for a in squad["data"]
                for p in a["paragraphs"])
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(squad, f, indent=2, ensure_ascii=False)
    print(f"[ok] wrote {out_path}")
    print(f"     passages={len(squad['data'])}  questions={n_qas}")


if __name__ == "__main__":
    main()
