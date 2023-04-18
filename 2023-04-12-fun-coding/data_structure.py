from wasabi import msg


def data_structure_design():
    problem = """
    Design a data structure that supports the following operations:
    void addWord(word)
    bool search(word)

    search(word) can search a literal word or a regual expression string
    containing only letters a-z or *. A * means it can represent any 
    one letter.

    Example:
    addWord("bad")
    addWord("dad")
    addWord("mad")
    search("pad") -> false
    search("bad") -> true
    search(".ad") -> true
    search("b..") -> true
    """

    class WordDictionary:
        # This is a trie problem

        def __init__(self):
            self.d = {}

        def addWord(self, word: str) -> None:
            # We want to add each character of the word in a nested dictionary
            cur = self.d
            if not self.search(word):
                for idx, char in enumerate(word):
                    if char not in cur:
                        cur[char] = {}
                    cur = cur[char]
                cur["[END]"] = {}

        def search(self, word: str) -> bool:
            cur = self.d

            def rec(start, cur):
                # Base condition
                if start >= len(word):
                    if "[END]" in cur:
                        return True
                    else:
                        return False

                if word[start] == ".":
                    for k, v in cur.items():
                        if rec(start + 1, v):
                            return True
                    return False
                elif word[start] in cur:
                    if rec(start + 1, cur[word[start]]):
                        return True
                else:
                    return False

            return rec(0, cur)


if __name__ == "__main__":
    data_structure_design()
