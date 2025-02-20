import random

from treelib import Tree
from typing import Optional
from tqdm import tqdm

def print_tree(tree: Tree, data_property: Optional[str]=None):
    tree_str = tree.show(stdout=False, data_property=data_property)
    tqdm.write(tree_str)


def example_tree():
    tree = Tree()
    tree.create_node("Node Root", "")  # root node
    tree.create_node("Node 0",    "0",    parent="")
    tree.create_node("Node 1",    "1",    parent="")
    tree.create_node("Node 00",   "00",   parent="0")
    tree.create_node("Node 01",   "01",   parent="0")
    tree.create_node("Node 10",   "10",   parent="1")
    tree.create_node("Node 100",  "100",  parent="10")
    tree.create_node("Node 101",  "101",  parent="10")
    tree.create_node("Node 1010", "1010", parent="101")
    return tree


def hard_example_tree():
    tree = Tree()
    tree.create_node("Node Root", "")  # root node
    tree.create_node("Node 0",    "0",    parent="")
    tree.create_node("Node 1",    "1",    parent="")
    tree.create_node("Node 2", "2", parent="")
    tree.create_node("Node 00",   "00",   parent="0")
    tree.create_node("Node 01",   "01",   parent="0")
    tree.create_node("Node 010", "010", parent="01")
    tree.create_node("Node 0100", "0100", parent="010")
    tree.create_node("Node 011", "011", parent="01")
    tree.create_node("Node 0110", "0110", parent="011")
    tree.create_node("Node 0111", "0111", parent="011")
    tree.create_node("Node 0112", "0112", parent="011")
    tree.create_node("Node 10",   "10",   parent="1")
    tree.create_node("Node 100",  "100",  parent="10")
    tree.create_node("Node 101",  "101",  parent="10")
    tree.create_node("Node 1010", "1010", parent="101")
    tree.create_node("Node 1011", "1011", parent="101")
    tree.create_node("Node 1012", "1012", parent="101")
    return tree


def jaxsonville_tree(num_floors: int = 2):
    tree = Tree()
    ## Ground Floor ##
    tree.create_node("Elevator Floor 1", "EL1")  # root node
    tree.create_node("Ground Floor", "G", parent="EL1")
    if num_floors == 1:
        return tree

    ## Floor 2 ##
    tree.create_node("Elevator Floor 2", "EL2", parent="EL1")
    tree.create_node("Main Hall Floor 2", "MH2", parent="EL2")
    tree.create_node("Main Hall Floor 2 North 1", "MH2N1", parent="MH2")
    tree.create_node("Room 210", "R210", parent="MH2N1")
    tree.create_node("Main Hall Floor 2 North 2", "MH2N2", parent="MH2N1")
    tree.create_node("North T Section Floor 2", "NT2", parent="MH2N2")
    tree.create_node("Room 205", "R205", parent="NT2")
    tree.create_node("North Hall Floor 2 West 1", "NH2W1", parent="NT2")
    tree.create_node("Room 204", "R204", parent="NH2W1")
    tree.create_node("North Hall Floor 2 West 2", "NH2W2", parent="NH2W1")
    tree.create_node("Room 203", "R203", parent="NH2W2")
    tree.create_node("North Hall Floor 2 West 3", "NH2W3", parent="NH2W2")
    tree.create_node("Room 201", "R201", parent="NH2W3")
    tree.create_node("Room 202", "R202", parent="NH2W3")
    tree.create_node("North Hall Floor 2 East 1", "NH2E1", parent="NT2")
    tree.create_node("Room 206", "R206", parent="NH2E1")
    tree.create_node("North Hall Floor 2 East 2", "NH2E2", parent="NH2E1")
    tree.create_node("Room 207", "R207", parent="NH2E2")
    tree.create_node("North Hall Floor 2 East 3", "NH2E3", parent="NH2E2")
    tree.create_node("Room 208", "R208", parent="NH2E3")
    tree.create_node("Room 209", "R209", parent="NH2E3")
    tree.create_node("Main Hall Floor 2 South 1", "MH2S1", parent="MH2")
    tree.create_node("Room 220", "R220", parent="MH2S1")
    tree.create_node("Main Hall Floor 2 South 2", "MH2S2", parent="MH2S1")
    tree.create_node("South T Section Floor 2", "ST2", parent="MH2S2")
    tree.create_node("Room 215", "R215", parent="ST2")
    tree.create_node("South Hall Floor 2 West 1", "SH2W1", parent="ST2")
    tree.create_node("Room 214", "R214", parent="SH2W1")
    tree.create_node("South Hall Floor 2 West 2", "SH2W2", parent="SH2W1")
    tree.create_node("Room 213", "R213", parent="SH2W2")
    tree.create_node("South Hall Floor 2 West 3", "SH2W3", parent="SH2W2")
    tree.create_node("Room 211", "R211", parent="SH2W3")
    tree.create_node("Room 212", "R212", parent="SH2W3")
    tree.create_node("South Hall Floor 2 East 1", "SH2E1", parent="ST2")
    tree.create_node("Room 216", "R216", parent="SH2E1")
    tree.create_node("South Hall Floor 2 East 2", "SH2E2", parent="SH2E1")
    tree.create_node("Room 217", "R217", parent="SH2E2")
    tree.create_node("South Hall Floor 2 East 3", "SH2E3", parent="SH2E2")
    tree.create_node("Room 218", "R218", parent="SH2E3")
    tree.create_node("Room 219", "R219", parent="SH2E3")

    if num_floors == 2:
        return tree
    """
    Computation Time = 670.23
    Traversal Time = 46, Tree Size = 42
    """

    ## Floor 3 ##
    tree.create_node("Elevator Floor 3", "EL3", parent="EL2")
    tree.create_node("Main Hall Floor 3", "MH3", parent="EL3")
    tree.create_node("Main Hall Floor 3 North 1", "MH3N1", parent="MH3")
    tree.create_node("Room 310", "R310", parent="MH3N1")
    tree.create_node("Main Hall Floor 3 North 2", "MH3N2", parent="MH3N1")
    tree.create_node("North T Section Floor 3", "NT3", parent="MH3N2")
    tree.create_node("Room 305", "R305", parent="NT3")
    tree.create_node("North Hall Floor 3 West 1", "NH3W1", parent="NT3")
    tree.create_node("Room 304", "R304", parent="NH3W1")
    tree.create_node("North Hall Floor 3 West 2", "NH3W2", parent="NH3W1")
    tree.create_node("Room 303", "R303", parent="NH3W2")
    tree.create_node("North Hall Floor 3 West 3", "NH3W3", parent="NH3W2")
    tree.create_node("Room 301", "R301", parent="NH3W3")
    tree.create_node("Room 302", "R302", parent="NH3W3")
    tree.create_node("North Hall Floor 3 East 1", "NH3E1", parent="NT3")
    tree.create_node("Room 306", "R306", parent="NH3E1")
    tree.create_node("North Hall Floor 3 East 2", "NH3E2", parent="NH3E1")
    tree.create_node("Room 307", "R307", parent="NH3E2")
    tree.create_node("North Hall Floor 3 East 3", "NH3E3", parent="NH3E2")
    tree.create_node("Room 308", "R308", parent="NH3E3")
    tree.create_node("Room 309", "R309", parent="NH3E3")
    tree.create_node("Main Hall Floor 3 South 1", "MH3S1", parent="MH3")
    tree.create_node("Room 320", "R320", parent="MH3S1")
    tree.create_node("Main Hall Floor 3 South 2", "MH3S2", parent="MH3S1")
    tree.create_node("South T Section Floor 3", "ST3", parent="MH3S2")
    tree.create_node("Room 315", "R315", parent="ST3")
    tree.create_node("South Hall Floor 3 West 1", "SH3W1", parent="ST3")
    tree.create_node("Room 314", "R314", parent="SH3W1")
    tree.create_node("South Hall Floor 3 West 2", "SH3W2", parent="SH3W1")
    tree.create_node("Room 313", "R313", parent="SH3W2")
    tree.create_node("South Hall Floor 3 West 3", "SH3W3", parent="SH3W2")
    tree.create_node("Room 311", "R311", parent="SH3W3")
    tree.create_node("Room 312", "R312", parent="SH3W3")
    tree.create_node("South Hall Floor 3 East 1", "SH3E1", parent="ST3")
    tree.create_node("Room 316", "R316", parent="SH3E1")
    tree.create_node("South Hall Floor 3 East 2", "SH3E2", parent="SH3E1")
    tree.create_node("Room 317", "R317", parent="SH3E2")
    tree.create_node("South Hall Floor 3 East 3", "SH3E3", parent="SH3E2")
    tree.create_node("Room 318", "R318", parent="SH3E3")
    tree.create_node("Room 319", "R319", parent="SH3E3")

    if num_floors == 3:
        return tree
    """
    Computation Time=1733.03
    Traversal Time=98, Tree Size=82
    """

    ## Floor 4 ##
    tree.create_node("Elevator Floor 4", "EL4", parent="EL3")
    tree.create_node("Main Hall Floor 4", "MH4", parent="EL4")
    tree.create_node("Main Hall Floor 4 North 1", "MH4N1", parent="MH4")
    tree.create_node("Room 410", "R410", parent="MH4N1")
    tree.create_node("Main Hall Floor 4 North 2", "MH4N2", parent="MH4N1")
    tree.create_node("North T Section Floor 4", "NT4", parent="MH4N2")
    tree.create_node("Room 405", "R405", parent="NT4")
    tree.create_node("North Hall Floor 4 West 1", "NH4W1", parent="NT4")
    tree.create_node("Room 404", "R404", parent="NH4W1")
    tree.create_node("North Hall Floor 4 West 2", "NH4W2", parent="NH4W1")
    tree.create_node("Room 403", "R403", parent="NH4W2")
    tree.create_node("North Hall Floor 4 West 3", "NH4W3", parent="NH4W2")
    tree.create_node("Room 401", "R401", parent="NH4W3")
    tree.create_node("Room 402", "R402", parent="NH4W3")
    tree.create_node("North Hall Floor 4 East 1", "NH4E1", parent="NT4")
    tree.create_node("Room 406", "R406", parent="NH4E1")
    tree.create_node("North Hall Floor 4 East 2", "NH4E2", parent="NH4E1")
    tree.create_node("Room 407", "R407", parent="NH4E2")
    tree.create_node("North Hall Floor 4 East 3", "NH4E3", parent="NH4E2")
    tree.create_node("Room 408", "R408", parent="NH4E3")
    tree.create_node("Room 409", "R409", parent="NH4E3")
    tree.create_node("Main Hall Floor 4 South 1", "MH4S1", parent="MH4")
    tree.create_node("Room 420", "R420", parent="MH4S1")
    tree.create_node("Main Hall Floor 4 South 2", "MH4S2", parent="MH4S1")
    tree.create_node("South T Section Floor 4", "ST4", parent="MH4S2")
    tree.create_node("Room 415", "R415", parent="ST4")
    tree.create_node("South Hall Floor 4 West 1", "SH4W1", parent="ST4")
    tree.create_node("Room 414", "R414", parent="SH4W1")
    tree.create_node("South Hall Floor 4 West 2", "SH4W2", parent="SH4W1")
    tree.create_node("Room 413", "R413", parent="SH4W2")
    tree.create_node("South Hall Floor 4 West 3", "SH4W3", parent="SH4W2")
    tree.create_node("Room 411", "R411", parent="SH4W3")
    tree.create_node("Room 412", "R412", parent="SH4W3")
    tree.create_node("South Hall Floor 4 East 1", "SH4E1", parent="ST4")
    tree.create_node("Room 416", "R416", parent="SH4E1")
    tree.create_node("South Hall Floor 4 East 2", "SH4E2", parent="SH4E1")
    tree.create_node("Room 417", "R417", parent="SH4E2")
    tree.create_node("South Hall Floor 4 East 3", "SH4E3", parent="SH4E2")
    tree.create_node("Room 418", "R418", parent="SH4E3")
    tree.create_node("Room 419", "R419", parent="SH4E3")

    if num_floors == 4:
        return tree
    """
    Computation Time=3428.89
    Traversal Time=159, Tree Size=122
    """

    ## Floor 5 ##
    tree.create_node("Elevator Floor 5", "EL5", parent="EL4")
    tree.create_node("Main Hall Floor 5", "MH5", parent="EL5")
    tree.create_node("Main Hall Floor 5 North 1", "MH5N1", parent="MH5")
    tree.create_node("Room 510", "R510", parent="MH5N1")
    tree.create_node("Main Hall Floor 5 North 2", "MH5N2", parent="MH5N1")
    tree.create_node("North T Section Floor 5", "NT5", parent="MH5N2")
    tree.create_node("Room 505", "R505", parent="NT5")
    tree.create_node("North Hall Floor 5 West 1", "NH5W1", parent="NT5")
    tree.create_node("Room 504", "R504", parent="NH5W1")
    tree.create_node("North Hall Floor 5 West 2", "NH5W2", parent="NH5W1")
    tree.create_node("Room 503", "R503", parent="NH5W2")
    tree.create_node("North Hall Floor 5 West 3", "NH5W3", parent="NH5W2")
    tree.create_node("Room 501", "R501", parent="NH5W3")
    tree.create_node("Room 502", "R502", parent="NH5W3")
    tree.create_node("North Hall Floor 5 East 1", "NH5E1", parent="NT5")
    tree.create_node("Room 506", "R506", parent="NH5E1")
    tree.create_node("North Hall Floor 5 East 2", "NH5E2", parent="NH5E1")
    tree.create_node("Room 507", "R507", parent="NH5E2")
    tree.create_node("North Hall Floor 5 East 3", "NH5E3", parent="NH5E2")
    tree.create_node("Room 508", "R508", parent="NH5E3")
    tree.create_node("Room 509", "R509", parent="NH5E3")
    tree.create_node("Main Hall Floor 5 South 1", "MH5S1", parent="MH5")
    tree.create_node("Room 520", "R520", parent="MH5S1")
    tree.create_node("Main Hall Floor 5 South 2", "MH5S2", parent="MH5S1")
    tree.create_node("South T Section Floor 5", "ST5", parent="MH5S2")
    tree.create_node("Room 515", "R515", parent="ST5")
    tree.create_node("South Hall Floor 5 West 1", "SH5W1", parent="ST5")
    tree.create_node("Room 514", "R514", parent="SH5W1")
    tree.create_node("South Hall Floor 5 West 2", "SH5W2", parent="SH5W1")
    tree.create_node("Room 513", "R513", parent="SH5W2")
    tree.create_node("South Hall Floor 5 West 3", "SH5W3", parent="SH5W2")
    tree.create_node("Room 511", "R511", parent="SH5W3")
    tree.create_node("Room 512", "R512", parent="SH5W3")
    tree.create_node("South Hall Floor 5 East 1", "SH5E1", parent="ST5")
    tree.create_node("Room 516", "R516", parent="SH5E1")
    tree.create_node("South Hall Floor 5 East 2", "SH5E2", parent="SH5E1")
    tree.create_node("Room 517", "R517", parent="SH5E2")
    tree.create_node("South Hall Floor 5 East 3", "SH5E3", parent="SH5E2")
    tree.create_node("Room 518", "R518", parent="SH5E3")
    tree.create_node("Room 519", "R519", parent="SH5E3")

    if num_floors == 5:
        return tree
    """
    Computation Time=5611.87
    Traversal Time=211, Tree Size=162
    """

    ## Floor 6 ##
    tree.create_node("Elevator Floor 6", "EL6", parent="EL5")
    tree.create_node("Main Hall Floor 6", "MH6", parent="EL6")
    tree.create_node("Main Hall Floor 6 North 1", "MH6N1", parent="MH6")
    tree.create_node("Room 610", "R610", parent="MH6N1")
    tree.create_node("Main Hall Floor 6 North 2", "MH6N2", parent="MH6N1")
    tree.create_node("North T Section Floor 6", "NT6", parent="MH6N2")
    tree.create_node("Room 605", "R605", parent="NT6")
    tree.create_node("North Hall Floor 6 West 1", "NH6W1", parent="NT6")
    tree.create_node("Room 604", "R604", parent="NH6W1")
    tree.create_node("North Hall Floor 6 West 2", "NH6W2", parent="NH6W1")
    tree.create_node("Room 603", "R603", parent="NH6W2")
    tree.create_node("North Hall Floor 6 West 3", "NH6W3", parent="NH6W2")
    tree.create_node("Room 601", "R601", parent="NH6W3")
    tree.create_node("Room 602", "R602", parent="NH6W3")
    tree.create_node("North Hall Floor 6 East 1", "NH6E1", parent="NT6")
    tree.create_node("Room 606", "R606", parent="NH6E1")
    tree.create_node("North Hall Floor 6 East 2", "NH6E2", parent="NH6E1")
    tree.create_node("Room 607", "R607", parent="NH6E2")
    tree.create_node("North Hall Floor 6 East 3", "NH6E3", parent="NH6E2")
    tree.create_node("Room 608", "R608", parent="NH6E3")
    tree.create_node("Room 609", "R609", parent="NH6E3")
    tree.create_node("Main Hall Floor 6 South 1", "MH6S1", parent="MH6")
    tree.create_node("Room 620", "R620", parent="MH6S1")
    tree.create_node("Main Hall Floor 6 South 2", "MH6S2", parent="MH6S1")
    tree.create_node("South T Section Floor 6", "ST6", parent="MH6S2")
    tree.create_node("Room 615", "R615", parent="ST6")
    tree.create_node("South Hall Floor 6 West 1", "SH6W1", parent="ST6")
    tree.create_node("Room 614", "R614", parent="SH6W1")
    tree.create_node("South Hall Floor 6 West 2", "SH6W2", parent="SH6W1")
    tree.create_node("Room 613", "R613", parent="SH6W2")
    tree.create_node("South Hall Floor 6 West 3", "SH6W3", parent="SH6W2")
    tree.create_node("Room 611", "R611", parent="SH6W3")
    tree.create_node("Room 612", "R612", parent="SH6W3")
    tree.create_node("South Hall Floor 6 East 1", "SH6E1", parent="ST6")
    tree.create_node("Room 616", "R616", parent="SH6E1")
    tree.create_node("South Hall Floor 6 East 2", "SH6E2", parent="SH6E1")
    tree.create_node("Room 617", "R617", parent="SH6E2")
    tree.create_node("South Hall Floor 6 East 3", "SH6E3", parent="SH6E2")
    tree.create_node("Room 618", "R618", parent="SH6E3")
    tree.create_node("Room 619", "R619", parent="SH6E3")

    if num_floors == 6:
        return tree
    """
    Computation Time = 8239.31
    Traversal Time = 264, Tree Size = 202
    """

    return tree


def adelphi_tree(num_floors: int = 2):
    tree = Tree()
    ## Ground Floor ##
    tree.create_node("Elevator Floor 1", "EL1")  # root node
    tree.create_node("Ground Floor", "G", parent="EL1")
    if num_floors == 1:
        return tree

    for floor in range(2, num_floors + 1):
        tree.create_node(f"Elevator Floor {floor}", f"EL{floor}", parent=f"EL{floor-1}")
        tree.create_node(f"Short Hall Floor {floor}", f"SH{floor}", parent=f"EL{floor}")
        tree.create_node(f"Corner Floor {floor}", f"C{floor}", parent=f"SH{floor}")
        tree.create_node(f"Main Hall 1 Floor {floor}", f"MH1F{floor}", parent=f"C{floor}")
        tree.create_node(f"Room {floor}{1:02}", f"R{floor}{1:02}", parent=f"MH1F{floor}")
        tree.create_node(f"Main Hall 2 Floor {floor}", f"MH2F{floor}", parent=f"MH1F{floor}")
        tree.create_node(f"Room {floor}{2:02}", f"R{floor}{2:02}", parent=f"MH2F{floor}")
        tree.create_node(f"Main Hall 3 Floor {floor}", f"MH3F{floor}", parent=f"MH2F{floor}")
        tree.create_node(f"Room {floor}{3:02}", f"R{floor}{3:02}", parent=f"MH3F{floor}")
        tree.create_node(f"Main Hall 4 Floor {floor}", f"MH4F{floor}", parent=f"MH3F{floor}")
        tree.create_node(f"Room {floor}{4:02}", f"R{floor}{4:02}", parent=f"MH4F{floor}")
        tree.create_node(f"Main Hall 5 Floor {floor}", f"MH5F{floor}", parent=f"MH4F{floor}")
        tree.create_node(f"Main Hall 6 Floor {floor}", f"MH6F{floor}", parent=f"MH5F{floor}")
        tree.create_node(f"Storage Room {floor}", f"SR{floor}", parent=f"MH6F{floor}")
        tree.create_node(f"Twin Room {floor}", f"TR{floor}", parent=f"MH6F{floor}")
        tree.create_node(f"Room {floor}{5:02}", f"R{floor}{5:02}", parent=f"TR{floor}")
        tree.create_node(f"Room {floor}{6:02}", f"R{floor}{6:02}", parent=f"TR{floor}")

    return tree


NUM_FLOORS = 4
MAX_HALLS_PER_FLOOR = 6
BRANCH_PROBABILITY = 0.3
MIN_HALL_LENGTH = 4
MAX_HALL_LENGTH = 7
ROOM_DENSITY = 0.5

def random_building_tree(num_floors: int = NUM_FLOORS,
                         max_halls_per_floor: int = MAX_HALLS_PER_FLOOR,
                         min_hall_length: int = MIN_HALL_LENGTH, max_hall_length: int = MAX_HALL_LENGTH,
                         room_density: float = ROOM_DENSITY):
    """
    Generate a random tree structure for a building.

    :param num_floors: Number of floors in the building.
    :param max_halls_per_floor: Maximum number of halls per floor.
    :param min_hall_length: Minimum length of each hall in terms of nodes.
    :param max_hall_length: Maximum length of each hall in terms of nodes.
    :param room_density: Probability of adding a room to each side of a hall node (0 to 1).
    :return: A tree representing the building.
    """
    tree = Tree()

    # Create the root node for the ground floor
    tree.create_node("Elevator Floor 1", "EL1")  # Root node

    for floor in range(1, num_floors + 1):
        if floor > 1:
            # Create elevator node for each floor
            tree.create_node(f"Elevator Floor {floor}", f"EL{floor}", parent=f"EL{floor - 1}")

        # Track the number of halls generated to enforce max_halls_per_floor
        hall_count = 1

        # Ensure exactly one hall comes out of the elevator
        hall_name = f"Hall Node {floor}-{hall_count}-0"
        hall_id = f"HN{floor}-{hall_count}-0"
        tree.create_node(hall_name, hall_id, parent=f"EL{floor}")

        # Generate a full hall chain
        def generate_hall_chain(parent_node):
            hall_count = int(parent_node.split('-')[1])
            hall_length = random.randint(min_hall_length, max_hall_length)
            current_node = parent_node
            for depth in range(1, hall_length):
                chain_name = f"Hall Node {floor}-{hall_count}-{depth}"
                chain_id = f"HN{floor}-{hall_count}-{depth}"
                tree.create_node(chain_name, chain_id, parent=current_node)
                current_node = chain_id

                # Optionally add rooms to the node based on room_density
                # Do not add at the end of the hall, as this can increase the max degree
                if random.random() < room_density and depth < hall_length - 1:
                    left_room_name = f"Left Room {floor}-{hall_count}-{depth}"
                    left_room_id = f"RL{floor}-{hall_count}-{depth}"
                    tree.create_node(left_room_name, left_room_id, parent=chain_id)
                if random.random() < room_density and depth < hall_length - 1:
                    right_room_name = f"Right Room {floor}-{hall_count}-{depth}"
                    right_room_id = f"RR{floor}-{hall_count}-{depth}"
                    tree.create_node(right_room_name, right_room_id, parent=chain_id)

                # Optionally branch new halls at the end of the chain
                if ((depth == hall_length - 1) and # Only branch at the end of the chain
                    (hall_count < max_halls_per_floor - 1) and # Limit number of halls per floor
                    (random.random() < BRANCH_PROBABILITY)): # Randomly decide if to branch or not
                    num_new_halls = 2
                    for new_hall in range(1, num_new_halls + 1):
                        hall_count += 1
                        new_hall_name = f"Hall Node {floor}-{hall_count}-{0}"
                        new_hall_id = f"HN{floor}-{hall_count}-{0}"
                        tree.create_node(new_hall_name, new_hall_id, parent=current_node)
                        hall_count = generate_hall_chain(new_hall_id)
            return hall_count

        # Start generating the hall chain
        generate_hall_chain(hall_id)

    return tree


def add_floors_to_tree(tree: Tree, floors_to_add: int, num_floors: int = NUM_FLOORS,
                       max_halls_per_floor: int = MAX_HALLS_PER_FLOOR,
                       min_hall_length: int = MIN_HALL_LENGTH, max_hall_length: int = MAX_HALL_LENGTH,
                       room_density: float = ROOM_DENSITY):
    """
    Generate a random tree structure for a building.

    :param num_floors: Number of floors in the building.
    :param max_halls_per_floor: Maximum number of halls per floor.
    :param min_hall_length: Minimum length of each hall in terms of nodes.
    :param max_hall_length: Maximum length of each hall in terms of nodes.
    :param room_density: Probability of adding a room to each side of a hall node (0 to 1).
    :return: A tree representing the building.
    """
    for floor in range(num_floors + 1, num_floors + floors_to_add + 1):
        if floor > 1:
            # Create elevator node for each floor
            tree.create_node(f"Elevator Floor {floor}", f"EL{floor}", parent=f"EL{floor - 1}")

        # Track the number of halls generated to enforce max_halls_per_floor
        hall_count = 1

        # Ensure exactly one hall comes out of the elevator
        hall_name = f"Hall Node {floor}-{hall_count}-0"
        hall_id = f"HN{floor}-{hall_count}-0"
        tree.create_node(hall_name, hall_id, parent=f"EL{floor}")

        # Generate a full hall chain
        def generate_hall_chain(parent_node):
            hall_count = int(parent_node.split('-')[1])
            hall_length = random.randint(min_hall_length, max_hall_length)
            current_node = parent_node
            for depth in range(1, hall_length):
                chain_name = f"Hall Node {floor}-{hall_count}-{depth}"
                chain_id = f"HN{floor}-{hall_count}-{depth}"
                tree.create_node(chain_name, chain_id, parent=current_node)
                current_node = chain_id

                # Optionally branch new halls at the end of the chain
                if ((depth == hall_length - 1) and  # Only branch at the end of the chain
                        (hall_count < max_halls_per_floor - 1) and  # Limit number of halls per floor
                        (random.random() < BRANCH_PROBABILITY)):  # Randomly decide if to branch or not
                    num_new_halls = 2
                    for new_hall in range(1, num_new_halls + 1):
                        hall_count += 1
                        new_hall_name = f"Hall Node {floor}-{hall_count}-{0}"
                        new_hall_id = f"HN{floor}-{hall_count}-{0}"
                        tree.create_node(new_hall_name, new_hall_id, parent=current_node)
                        hall_count = generate_hall_chain(new_hall_id)
                else:
                    # Optionally add rooms to the node based on room_density
                    if random.random() < room_density:
                        left_room_name = f"Left Room {floor}-{hall_count}-{depth}"
                        left_room_id = f"RL{floor}-{hall_count}-{depth}"
                        tree.create_node(left_room_name, left_room_id, parent=chain_id)
                    if random.random() < room_density:
                        right_room_name = f"Right Room {floor}-{hall_count}-{depth}"
                        right_room_id = f"RR{floor}-{hall_count}-{depth}"
                        tree.create_node(right_room_name, right_room_id, parent=chain_id)

            return hall_count

        # Start generating the hall chain
        generate_hall_chain(hall_id)

    return tree


def stretch_halls(tree: Tree, hall_length_to_add: int, room_density: float = ROOM_DENSITY):
    """
    Extends each hall in the tree by hall_length_to_add, adding rooms with given density.
    Ensures that halls are stretched without introducing splits.
    """

    for node in list(tree.nodes.values()):
        if node.identifier.startswith("HN"):
            floor, hall_count, depth = node.identifier[2:].split('-')
            # Identify hall nodes
            hall_nodes = [n for n in tree.nodes.values() if n.identifier.startswith(f"HN{floor}-{hall_count}-")]
            # Find the last extended node of the hall
            last_extended_hall = max(hall_nodes, key=lambda n: int(n.identifier.split('-')[-1]))
            last_depth = int(last_extended_hall.identifier.split('-')[-1])
            parent_id = last_extended_hall.identifier

            for depth in range(last_depth + 1, last_depth + 1 + random.randint(0, hall_length_to_add)):
                chain_name = f"Hall Node {floor}-{hall_count}-{depth}"
                chain_id = f"HN{floor}-{hall_count}-{depth}"

                if chain_id in tree.nodes:
                    parent_id = chain_id  # Continue from existing extended node
                    continue  # Skip creating duplicate nodes

                tree.create_node(chain_name, chain_id, parent=parent_id)
                parent_id = chain_id

                if random.random() < room_density:
                    tree.create_node(f"Left Room {floor}-{hall_count}-{depth}", f"RL{floor}-{hall_count}-{depth}", parent=chain_id)
                if random.random() < room_density:
                    tree.create_node(f"Right Room {floor}-{hall_count}-{depth}", f"RR{floor}-{hall_count}-{depth}", parent=chain_id)

    return tree


def increase_room_density(tree: Tree, room_density_to_add: float):
    """
    Increases the number of rooms along each hall by sampling additional rooms based on density.
    """
    for node in list(tree.nodes.values()):
        if node.identifier.startswith("HN"):
            suffix = node.identifier[2:]
            if f"RL{suffix}" not in tree.nodes and random.random() < room_density_to_add:
                tree.create_node(f"Room {node.identifier}-L", f"RL{suffix}", parent=node.identifier)
            if f"RR{suffix}" not in tree.nodes and random.random() < room_density_to_add:
                tree.create_node(f"Room {node.identifier}-R", f"RR{suffix}", parent=node.identifier)

    return tree