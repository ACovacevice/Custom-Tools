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
    
    def __init__(self, **kwargs) -> None:
        """
        Args:
            pattern1, pattern2, ... (str): 
                Regex patterns to lookout for (first is mandatory, rest is optional).
            repl (str): 
                Replacement value for matches.
                
        Obs.: One important thing is to encapsulate patterns within parentheses; otherwise, it may not work.        
        """
        kwargs.update({k: re.compile(v, re.IGNORECASE) for k, v in kwargs.items() if k.startswith("pattern")})
        self.__dict__ = kwargs
        self.__pats = [kwargs[k] for k in sorted(kwargs.keys()) if k.startswith("pattern")]
        self.__pos = 0

    
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
