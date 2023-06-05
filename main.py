import argparse
import random
import time


import computergym
import gym
from llm_agent import LLMAgent

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains

import logging

import urllib3
urllib3.disable_warnings() # disable http warning when closing env
import warnings
warnings.filterwarnings("ignore") #remove Userwarning

logging.basicConfig(level=logging.INFO)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="click-button")
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--llm", type=str, default="chatgpt")
    parser.add_argument("--erci", type=int, default=0)
    parser.add_argument("--step", type=int, default=-1)
    parser.add_argument("--irci", type=int, default=1)
    parser.add_argument("--sgrounding", action="store_true", default=False)
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--prompt-token-price", type=float, default=0.002)
    parser.add_argument("--completion-token-price", type=float, default=0.002)
    opt = parser.parse_args()

    return opt


def web(opt, url):
    driver = get_webdriver(url)

    while True:
        llm_agent = LLMAgent(
            opt.env, rci_plan_loop=opt.erci, rci_limit=opt.irci, llm=opt.llm
        )

        html_body = get_html_state_from_real(driver, opt)

        llm_agent.update_html_state(html_body)

        # Set objective (e.g., login with id and pw)
        goal = input("Type your command (type 'exit' to quit): ")
        if goal == "exit":
            break
        llm_agent.set_goal(goal)

        llm_agent.initialize_plan()

        step = llm_agent.get_plan_step()
        logging.info(f"The number of generated action steps: {step}")
        for _ in range(step):
            instruction = llm_agent.generate_action()
            logging.info(f"Instruction: {instruction}")
            perform_instruction(driver, instruction)

            html_body = get_html_state_from_real(driver, opt)
            llm_agent.update_html_state(html_body)

    driver.quit()


def get_html_state_from_real(driver, opt):
    if opt.env == "facebook":
        main_html_xpath = '//*[@id="content"]'
        html_body = driver.find_element(By.XPATH, main_html_xpath).get_attribute(
            "outerHTML"
        )
    else:
        raise NotImplemented

    return html_body


def perform_instruction(driver, instruction):
    instruction = instruction.split(" ")
    inst_type = instruction[0]
    inst_type = inst_type.lower()

    if inst_type == "type":
        characters = " ".join(instruction[1:])
        characters = characters.replace('"', "")
        chain = ActionChains(driver)
        chain.send_keys(characters)
        chain.perform()
    elif inst_type == "clickxpath":
        xpath = " ".join(instruction[1:])
        element = driver.find_element(By.XPATH, str(xpath))
        chain = ActionChains(driver)
        chain.move_to_element(element).click().perform()
    elif inst_type == "press":
        key_type = instruction[1]
        # TODO: press special key
        if key_type == "enter":
            chain = ActionChains(driver)
            chain.send_keys("\n")
            chain.perform()
        elif key_type == "space":
            chain = ActionChains(driver)
            chain.send_keys(" ")
            chain.perform()
        else:
            raise NotImplemented
    else:
        raise ValueError("Invalid instruction")


def get_webdriver(url):
    options = webdriver.ChromeOptions()
    # options.add_argument("headless")
    options.add_argument("disable-gpu")
    options.add_argument("no-sandbox")

    driver = webdriver.Chrome(chrome_options=options)
    driver.implicitly_wait(5)
    driver.maximize_window()
    driver.implicitly_wait(5)

    driver.get(url)
    driver.implicitly_wait(10)
    return driver


def miniwob(opt):
    env = gym.make("MiniWoBEnv-v0", env_name=opt.env, headless=opt.headless)
    time.sleep(10) #wait for the env to load
    if any(not instance.is_alive() for instance in env.instances): #TODO understand why its possible to run multiple Gym instances 
        raise Exception("Environment has crashed : Wrong MINIWOB_BASE_URL or unknown task ?")
    success = 0
    number_of_token_sent_per_episode = []
    number_of_token_received_per_episode = []
    number_of_calls_per_episode = []
    time_taken_per_episode = []
    config_string = (
        f"state_{opt.sgrounding}-erci_{opt.erci}-irci_{opt.irci}"
    )
    exp_path = f"history/"+ opt.llm + "/" + opt.env +  "/" + config_string + "/" + time.strftime("%Y%m%d-%H%M%S")

    for _ in range(opt.num_episodes):
        #measure time taken 
        info = ""
        start_time = time.time()

        llm_agent = LLMAgent(
            opt.env,
            rci_plan_loop=opt.erci,
            rci_limit=opt.irci,
            llm=opt.llm,
            state_grounding=opt.sgrounding,
            exp_path = exp_path,
        )
        # initialize environment
        states = env.reset(seeds=[random.random()], record_screenshots=True)
        llm_agent.set_goal(states[0].utterance)
        html_state = get_html_state(opt, states)

        llm_agent.update_html_state(html_state)

        try:
            llm_agent.initialize_plan()
        except Exception as e:
            llm_agent.save_error( str(e) + "\n") 
            continue

        if opt.step == -1:
            step = llm_agent.get_plan_step()
        else:
            step = opt.step

        logging.info(f"The number of generated action steps: {step}")
        for _ in range(step):
            assert len(states) == 1
            try:
                instruction = llm_agent.generate_action()
                logging.info(f"The executed instruction: {instruction}")

                miniwob_action = llm_agent.convert_to_miniwob_action(instruction)

                states, rewards, dones, _ = env.step([miniwob_action])
                llm_agent.save_action(str(miniwob_action))
            except Exception as e:
                info += str(e) + "\n"
                llm_agent.save_action(str(e))
                rewards = [0]
                dones = [True]
                break

            if rewards[0] != 0:
                break

            if all(dones):  # or llm_agent.check_finish_plan():
                break

            html_state = get_html_state(opt, states)
            llm_agent.update_html_state(html_state)

        if rewards[0] > 0:
            success += 1
            llm_agent.save_result(True)
        else:
            llm_agent.save_result(False, cause = info)
        

        number_of_token_sent_per_episode.append(llm_agent.number_of_token_sent)
        number_of_token_received_per_episode.append(llm_agent.number_of_token_received)
        number_of_calls_per_episode.append(llm_agent.number_of_calls)
        time_taken_per_episode.append(time.time() - start_time)

    assert len(number_of_token_sent_per_episode) == opt.num_episodes

    success_rate = success / opt.num_episodes
    print(f"success rate: {success_rate}")
    env.close()

    result_dict = {
        "success_rate": success_rate,
        "min_sent": min(number_of_token_sent_per_episode),
        "max_sent": max(number_of_token_sent_per_episode),
        "mean_sent": sum(number_of_token_sent_per_episode) / len(number_of_token_sent_per_episode),
        "min_received": min(number_of_token_received_per_episode),
        "max_received": max(number_of_token_received_per_episode),
        "mean_received": sum(number_of_token_received_per_episode) / len(number_of_token_received_per_episode),
        "mean_calls": sum(number_of_calls_per_episode) / len(number_of_calls_per_episode),
        "time": sum(time_taken_per_episode) / len(time_taken_per_episode),
        "cost" :  sum(number_of_token_sent_per_episode)/1000*opt.prompt_token_price +  sum(number_of_token_received_per_episode)/1000*opt.completion_token_price, #hardcoded chatgpt price
        "experiment folder" : exp_path
    }

    return result_dict


def get_html_state(opt, states):
    extra_html_task = [
        "click-dialog",
        "click-dialog-2",
        "use-autocomplete",
        "choose-date",
    ]

    html_body = states[0].html_body
    if opt.env in extra_html_task:
        html_body += states[0].html_extra
    return html_body


if __name__ == "__main__":
    opt = parse_opt()
    if opt.env == "facebook":
        url = "https://www.facebook.com/"
        web(opt, url)
    else:
        print(opt.headless)
        miniwob(opt)
