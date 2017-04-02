# -*- coding: utf-8 -*-
from nltk.corpus import stopwords


def get_augmented_stopwords():
    stop_words = set(stopwords.words("english"))
    df_identified_stopwords = [",", ".", "(", ")", ":", "'s", "-", "inform",
            "use","servic", "one", "provid", "work", "''","time","``", "make",\
            "may", "also", "area", "includ", "year", "contact", ";", "need", \
            "&", "?", "new","avail", "us", "help", "canada", "experi", "home",\
            "first", "day", "develop", "site", "program", "person", "like",\
            "well", "univers", "follow","offer", "take", "page", "two", \
            "last", "get", "!", "plan", "guid", "report", "mani", "system",\
            "number", "activ", "commun", "locat", "link", "part", "peopl",\
            "see", "requir", "resourc", "nation", "employ", "gener", "term",\
            "student", "respons", "addit", "best", "world", "manag", "right",\
            "within", "special", "group", "state", "research", "list", "look",\
            "place", "travel", "organ", "...", "opportun", "chang", "form",\
            "local", "job", "network", "2", "associ", "good", "educ", \
            "public", "come", "waterloo", "condit", "set", "sever", \
            "would", "1", "differ", "river", "way", "career", "call", \
            "find", "current", "great", "import", "give", "canadian", "free",\
            "level", "pleas", "copyright", "back", "must", "search", "trip", \
            "want", "major", "rate", "becom", "north", "interest", "meet", \
            "possibl", "2001", "@", "environ", "result", "learn", "high", \
            "three", "depart", "|", "keep", "hour", "receiv", "continu", \
            "much", "support", "start", "maintain", "10", "howev", "consid", \
            "water", "present", "know", "winter", "go", "even", "intern", \
            "centr", "tri", "3", "detail", "n't", "point", "everi", "direct", 
            "check", "cover", "uw", "co-op", "larg", "countri", "begin", \
            "skill", "futur", "mountain", "staff", "end", "region", "feel", \
            "specif", "date", "event", "type", "updat", "fall", "reserv", 
            "request", "rang", "5", "relat", "month", "design", "oper", 
            "summer", "next", "web", "studi", "run","]","["]
    for new_stpwrd in df_identified_stopwords:
        stop_words.add(new_stpwrd)
    return stop_words