from colorama import Fore, Style

def log_info(message):
    print(f"{Fore.BLUE}[INFO]{Style.RESET_ALL} {message}")

def log_info_node(id, message):
    print(f"{Fore.MAGENTA}[Node {id}]{Style.RESET_ALL} {message}")

def log_success(message):
    print(f"{Fore.GREEN}[SUCCESS]{Style.RESET_ALL} {message}")

def log_debug(message):
    print(f"{Fore.MAGENTA}[SUCCESS]{Style.RESET_ALL} {message}")

def log_warning(message):
    print(f"{Fore.YELLOW}[WARNING]{Style.RESET_ALL} {message}")

def log_error(message):
    print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} {message}")