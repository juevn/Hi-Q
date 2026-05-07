from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Union
from functools import cached_property
from collections import defaultdict
import os
import json

from .base_dataset import BaseDataset
from .registry import register_dataset

PathLike = Union[str, Path]
JsonDict = Dict[str, Any]


@register_dataset("hotpotqa_full")
@register_dataset("hotpotqa")
class HotpotQA(BaseDataset):
    """HotpotQA dataset class
    data format:
    [
        {
            "_id": "5abe953b5542993f32c2a170",
            "answer": "superhero roles as the Marvel Comics",
            "question": "what is one of the stars of  The Newcomers known for",
            "supporting_facts": [
                [
                    "The Newcomers (film)",
                    0
                ],
                [
                    "Chris Evans (actor)",
                    1
                ]
            ],
            "context": [
                [
                    "Vaada Poda Nanbargal",
                    [
                        "Vaada Poda Nanbargal is a 2011 Indian Tamil-language romantic comedy film directed by Manikai.",
                        " P. Arumaichandran has produced this movie under the banner 8 Point Entertainments.",
                        " The film stars newcomers Nanda, Sharran Kumar and Yashika in the lead roles.",
                        " The lead actor Nanda happens to be one of the strong contender of a popular television series \"Yaar Adutha Prabhu Deva\" aired on Vijay TV."
                    ]
                ],
                [
                    "Chris Evans (actor)",
                    [
                        "Christopher Robert Evans (born June 13, 1981) is an American actor and filmmaker.",
                        " Evans is known for his superhero roles as the Marvel Comics characters Steve Rogers / Captain America in the Marvel Cinematic Universe and Johnny Storm / Human Torch in \"Fantastic Four\" and ."
                    ]
                ],
                [
                    "NSYNC in Concert",
                    [
                        "NSYNC in Concert (also known as the Second II None Tour, Ain't No Stoppin' Us Now Tour, Boys of Summer Tour and The Winter Shows) is the second concert tour by American boy band, NSYNC.",
                        " Primarily visiting North America, the tour supported the band's debut studio album, \"NSYNC\".",
                        " The trek lasted eighteen months, playing over two hundred concerts in over one hundred cities.",
                        " In 1998, the tour was nominated for \"Best New Artist Tour\" by Pollstar Concert Industry Awards.",
                        " It also became one of the biggest tours in 1999, earning over $50 million.",
                        " Supporting the band on the tour were newcomers Britney Spears, B*Witched and Mandy Moore along with music veterans Jordan Knight, Shanice and The Sugarhill Gang."
                    ]
                ],
                [
                    "The Awakening (P.O.D. album)",
                    [
                        "Doug Van Pelt, giving the album four stars for \"HM Magazine\", writes, \"When looking back at P.O.D.\u2019s amazing career, there\u2019s probably going to be some landmark albums that stand out in most fan\u2019s minds... This one is not far behind.\"",
                        " Awarding the album four out of five stars from \"CCM Magazine\", Matt Conner states, they will \"remain at the top\", with Howard Benson's production, where it has the theme of a \"central character coming to terms with his own mistakes...[giving the album] meaningful depth.\"",
                        " Mary Nikkel, rating the album four and a half stars at New Release Today, says, \"\"The Awakening\" easily one of the strongest rock releases of the year, a must-have for longtime fans and newcomers thirsty for some heavy music with substance.\"",
                        " Indicating in a ten out of ten review at Cross Rhythms, replies, \"An album likely to be eventually acknowledged as P.O.D.'s finest ever release.\"",
                        " Chad Bowar, rating the album three stars from About.com, says, \"P.O.D. fans will be intrigued by the wide variety of styles and the memorable songs on \"The Awakening\".\""
                    ]
                ],
                [
                    "Tere Mere Phere",
                    [
                        "\"Tere Mere Phere\" (Hindi: \u0924\u0947\u0930\u0947 \u092e\u0947\u0930\u0947 \u092b\u0947\u0930\u0947 English: Our wedding vows ) is an 2011 Hindi romantic comedy, Road film directed by well known and respected actress Deepa Sahi, and produced by the internationally acclaimed producer director Ketan Mehta and renowned singer Anup Jalota.",
                        " Presented by Sitara Productions, it is a Maya Movies Production.",
                        " It stars Vinay Pathak.",
                        " Riya Sen in the lead roles and introduces newcomers Jagrat Desai and Sasha Goradia.",
                        " The story of the film has been written by Deepa Sahi and Jagrat Desai.",
                        " The music is by Shivi R. Kashyap and the lyrics are penned by Manoj \u2018Muntashir\u2019 and Ketan Mehta.",
                        " Most of the songs are choreographed by Bollywood choreographer Jeet Singh.",
                        " It was released on 30 September 2011 to a mixed but mostly positive reception from critics."
                    ]
                ],
                [
                    "Easan",
                    [
                        "Easan (Tamil: \u0b88\u0b9a\u0ba9\u0bcd ) is a 2010 Indian Tamil-language drama film written, directed and produced by M. Sasikumar, directing his second film after the blockbuster, \"Subramaniapuram\".",
                        " It stars Samuthirakani, Vaibhav, producer A. L. Alagappan and Abhinaya in lead roles alongside several newcomers.",
                        " The film was known and referred to as \"Nagaram\" and \"Aaga Chiranthavan\" before the official title was confirmed.",
                        " It released on 17 December 2010."
                    ]
                ],
                [
                    "Nan Love Track",
                    [
                        "Nan Love Track (Kannada: \u0ca8\u0ca8\u0ccd \u0cb2\u0cb5\u0ccd \u0c9f\u0ccd\u0cb0\u0ccd\u0caf\u0cbe\u0c95\u0ccd ) is a 2016 Indian Kannada language romance film directed by Kathir, who is best known for his successful Tamil films such as \"Kadhal Desam\" (1996) and \"Kadhalar Dhinam\"(1999) and \"Idhayam\"(1991), making his debut in Kannada cinema.",
                        " The film stars newcomers Rakshith Gowda and Nidhi Kushalappa in the lead roles."
                    ]
                ],
                [
                    "The Newcomers (film)",
                    [
                        "The Newcomers is a 2000 American family drama film directed by James Allen Bradley and starring Christopher McCoy, Kate Bosworth, Paul Dano and Chris Evans.",
                        " Christopher McCoy plays Sam Docherty, a boy who moves to Vermont with his family, hoping to make a fresh start away from the city.",
                        " It was filmed in Vermont, and released by Artist View Entertainment and MTI Home Video."
                    ]
                ],
                [
                    "Sandra Dee",
                    [
                        "Sandra Dee (born Alexandra Zuck; April 23, 1942\u00a0\u2013 February 20, 2005) was an American actress.",
                        " Dee began her career as a child model, working in commercials before transitioning to film in her teenage years.",
                        " Best known for her portrayal of ing\u00e9nues, Dee earned a Golden Globe Award as one of the year's most promising newcomers for her performance in Robert Wise's \"Until They Sail\" (1958).",
                        " She became a teenage star for her subsequent performances in \"Imitation of Life\" and \"Gidget\" (both 1959), which made her a household name."
                    ]
                ],
                [
                    "Dan Kavanagh",
                    [
                        "Dan Kavanagh is a British rock drummer best known for his work with Jamie Lenman and Godsized.",
                        " In May 2014 he was listed as one of the top 10 British drumming newcomers by Rhythm, who called him a \"hard hitting rock fiend juggling two intense gigs\".",
                        " He has since become a contributor to Rhythm."
                    ]
                ]
            ],
            "type": "bridge",
            "level": "hard"
        },
        ...
    ]
    """

    def __init__(self, cfg):
        benchmark_conf = cfg.benchmark

        self.corpus_path = Path(benchmark_conf.corpus_path)
        if not self.corpus_path.exists():
            raise FileNotFoundError(f"Corpus not found: {self.corpus_path}")
        with open(self.corpus_path, "r", encoding="utf-8") as f:
            corpus = json.load(f)
        self.doc_corpus = [f"{doc['title']}\n{doc['text']}" for doc in corpus]

        self.dataset_path = Path(benchmark_conf.dataset_path)
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            datas = json.load(f)

        for data in datas:
            data["id"] = data["_id"]

        super().__init__(datas)
