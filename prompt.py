import os


class Prompt:
    def __init__(self, env: str = "click-button") -> None:
        self.llm = "davinci"

        if "webshop" in env:
            base_dir = f"prompt/webshop/"  ## put webshop relevant files here
            self.searchbar_regex = "(.+)\[(.+)\]"
            def check_regex(self, instruciton):  
                return not re.search(self.prompt.searchbar_regex, instruciton, flags=re.I)
        
        else :
            self.davinci_type_regex = "^type\s.{1,}$"
            self.chatgpt_type_regex = '^type\s[^"]{1,}$'
            self.press_regex = ("^press\s(enter|arrowleft|arrowright|arrowup|arrowdown|backspace)$")
            self.clickxpath_regex = "^clickxpath\s.{1,}$"
            self.clickoption_regex = "^clickoption\s.{1,}$"
            self.movemouse_regex = "^movemouse\s.{1,}$"

            def check_regex(self, instruciton):  
                return (
                    (not re.search(self.prompt.clickxpath_regex, instruciton, flags=re.I))
                    and (not re.search(self.prompt.chatgpt_type_regex, instruciton, flags=re.I))
                    and (not re.search(self.prompt.davinci_type_regex, instruciton, flags=re.I))
                    and (not re.search(self.prompt.press_regex, instruciton, flags=re.I))
                    and (not re.search(self.prompt.clickoption_regex, instruciton, flags=re.I))
                    and (not re.search(self.prompt.movemouse_regex, instruciton, flags=re.I))
                )
        self.check_regex = check_regex

            elif os.path.exists(f"prompt/{env}/"):
                base_dir = f"prompt/{env}/"
            else:
                base_dir = f"prompt/"

        with open(base_dir + "example.txt") as f:
            self.example_prompt = f.read()

        with open(base_dir + "first_action.txt") as f:
            self.first_action_prompt = f.read()

        with open(base_dir + "base.txt") as f:
            self.base_prompt = f.read()
            self.base_prompt = self.replace_regex(self.base_prompt)

        with open(base_dir + "initialize_plan.txt") as f:
            self.init_plan_prompt = f.read()

        with open(base_dir + "action.txt") as f:
            self.action_prompt = f.read()

        with open(base_dir + "rci_action.txt") as f:
            self.rci_action_prompt = f.read()
            self.rci_action_prompt = self.replace_regex(self.rci_action_prompt)

        with open(base_dir + "update_action.txt") as f:
            self.update_action = f.read()

    def replace_regex(self, base_prompt):
        if self.llm == "chatgpt":
            base_prompt = base_prompt.replace("{type}", self.chatgpt_type_regex)
        elif self.llm == "davinci":
            base_prompt = base_prompt.replace("{type}", self.davinci_type_regex)
        else:
            raise NotImplemented

        base_prompt = base_prompt.replace("{press}", self.press_regex)
        base_prompt = base_prompt.replace("{clickxpath}", self.clickxpath_regex)
        base_prompt = base_prompt.replace("{clickoption}", self.clickoption_regex)
        base_prompt = base_prompt.replace("{movemouse}", self.movemouse_regex)

        return base_prompt

    def get_reverse_dict(self):
        prompt_texts = {}

        for attr_name in dir(self):
            attr_value = getattr(self, attr_name)
            if isinstance(attr_value, str) and len(attr_value) > 0:
                prompt_texts[attr_value] = "[" + attr_name.upper() + "]"

        return prompt_texts
