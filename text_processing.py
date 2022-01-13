import re

class DeepSub:
    
    """
    Class for recursive string replacement using multiple regex patterns.
    
    Example:
    
        >> text = 'O rato roeu a roupa do rei de Roma.'
        >> cls = DeepSub(pattern1=r'(roupa)', pattern2=r'([aeiou])', repl=r"-")
        >> result = cls.sub(text)
        >> print(result)
        
        Output:
            'O rato roeu a r--p- do rei de Roma.'
        
        The first pattern found a match for the word 'roupa'. 
        The second pattern found vowels inside the word 'roupa'.
        Replacement for the final matches took place, replacing vowels in 'roupa' by '-'.
    """
    
    def __init__(self, repl: str, flags: int = 0, **kwargs) -> None:
        """
        Args:
            pattern1, pattern2, ... (str): 
                Regex patterns to lookout for (first is mandatory, rest is optional).
            repl (str): 
                Replacement value for matches.
            flags (int, optional):
                Flags to be passed to the regex engine. Defaults to 0.
        Obs.: One important thing is to encapsulate patterns within parentheses; otherwise, it may not work.        
        """
        kwargs.update({k: re.compile(v, flags) for k, v in kwargs.items() if k.startswith("pattern")})
        self.__dict__ = kwargs
        self.__pats = [kwargs[k] for k in sorted(kwargs.keys()) if k.startswith("pattern")]
        self.__pos = 0
        self.repl = repl

    
    def __sub(self, match: re.Match) -> str:
        self.__pos += 1
        if self.__pos == len(self.__pats) - 1:
            self.__pos = 0
            return self.__pats[-1].sub(self.repl, match.groups()[0])        
        return self.__pats[self.__pos].sub(self.__sub, match.groups()[0])


    def sub(self, text: str) -> str:
        if len(self.__pats) == 1:
            return self.__pats[0].sub(self.repl, text)
        return self.__pats[self.__pos].sub(self.__sub, text)


def reformat_abbreviations(text: str) -> str:
    pattern = DeepSub(pattern1="((:?[A-Z]+\.)+)", pattern2=r"(\.)", repl="")
    return pattern.sub(text)


def replace_email(text: str, by: str = " ") -> str:
    pattern = re.compile(
        r"([A-Z0-9_.+-]+@[A-Z0-9-]+(?:\.[A-Z0-9-]+)+)", 
        flags=2
    )
    return pattern.sub(by, text)


def replace_url(text: str, by: str = " ") -> str:
    pattern = re.compile(
        r"(https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*))", 
        flags=2
    )
    return pattern.sub(by, text)


def replace_equation(text: str, by: str = "equacao") -> str:

    def reduce_args(text):
        cls = DeepSub(pattern1=r"(\([^\(\)]+\)|\{[^\{\}]+\}|\[[^\[\]]+\])+", repl="ARGS", flags=2)
        result = cls.sub(text)
        if result == text:
            return text
        return reduce_args(result)

    def reduce_prod(text):
        cls = DeepSub(pattern1=r"((:?[a-z0-9]+\s*[\*\^]+\s*[a-z0-9]+)+)", repl="PROD", flags=2)
        result = cls.sub(text)
        if result == text:
            return text
        return reduce_prod(result)

    def reduce_div(text):
        cls = DeepSub(pattern1=r"((:?[a-z0-9]+\s*[\/]+\s*[a-z0-9]+)+)", repl="DIV", flags=2)
        result = cls.sub(text)
        if result == text:
            return text
        return reduce_div(result)

    def reduce_add(text):
        cls = DeepSub(pattern1=r"((:?[a-z0-9]+\s*[\+\-]+\s*[a-z0-9]+)+)", repl="ADD", flags=2)
        result = cls.sub(text)
        if result == text:
            return text
        return reduce_add(result)

    def reduce_equation(text):
        cls = DeepSub(pattern1=r"((:?[a-z0-9]+\s*[><=]+\s*[a-z0-9]+)+)", repl=by, flags=2)
        result = cls.sub(text)
        if result == text:
            return text
        return reduce_equation(result)
    
    red_text = reduce_equation(reduce_add(reduce_div(reduce_prod(reduce_args(text)))))
    
    if by in red_text:
        return red_text
    
    return text
