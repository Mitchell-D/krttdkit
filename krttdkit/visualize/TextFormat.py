from copy import deepcopy

class TextFormat:
    """
    Extremely basic static class for applying ANSII escape color codes to
    strings for expressive terminal printing. the class is called with one
    of its built-in colors and optional boolean modifiers as keyword arguments.

    For example:

    print(TextFormat.RED("WHAT?!", bold=True, bright=True))
    """
    kw = {
            "bold":False,
            "highlight":False,
            "bright":False,
            "underline":False,
            }
    @staticmethod
    def test():
        kwargs = [{"normal":True},{"bright":True},
                  {"bold":True},{"highlight":True},
                  {"underline":True}]
        fullstr = []
        for i in range(len(kwargs)):
            linestr = ["normal", "bright", "bold", "highlight", "underline"]
            printstr = " ".join(
                    [f(linestr[i], **{linestr[i]:True}) for f in
                     [TextFormat.WHITE, TextFormat.RED, TextFormat.GREEN,
                      TextFormat.YELLOW, TextFormat.BLUE, TextFormat.PURPLE,
                      TextFormat.CYAN, TextFormat.BLACK]])
            fullstr.append(printstr)
        return "\n".join(fullstr)


    @staticmethod
    def _get_color(color:int, text:str, **kwargs):
        tmp_kw = deepcopy(TextFormat.kw)
        tmp_kw.update(kwargs)
        cstr = str(["3","9"][tmp_kw['bright']]) + str(color)
        if tmp_kw["highlight"]:
            fstr = ";7m"
        elif tmp_kw["bold"]:
            fstr = ";1m"
        elif tmp_kw["underline"]:
            fstr = ";4m"
        else:
            cstr += "m"
            fstr = ""
        return f"\033[{cstr}{fstr}{text}\033[0m".strip()

    @staticmethod
    def BLACK(text:str, **kwargs):
        return TextFormat._get_color(0, text, **kwargs)

    @staticmethod
    def RED(text:str, **kwargs):
        return TextFormat._get_color(1, text, **kwargs)

    @staticmethod
    def GREEN(text:str, **kwargs):
        return TextFormat._get_color(2, text, **kwargs)

    @staticmethod
    def YELLOW(text:str, **kwargs):
        return TextFormat._get_color(3, text, **kwargs)

    @staticmethod
    def BLUE(text:str, **kwargs):
        return TextFormat._get_color(4, text, **kwargs)

    @staticmethod
    def PURPLE(text:str, **kwargs):
        return TextFormat._get_color(5, text, **kwargs)

    @staticmethod
    def CYAN(text:str, **kwargs):
        return TextFormat._get_color(6, text, **kwargs)

    @staticmethod
    def WHITE(text:str, **kwargs):
        return TextFormat._get_color(7, text, **kwargs)

