import os
import shutil
import numpy

def copy_gw_event(GW_event: str):
    og_path = f"../{GW_event}/" # one dir up
    new_path = f"./{GW_event}/" # start from this dir
    for string in ["bns", "nsbh", "default"]:
        result_file = os.path.join(og_path, string, f"{string}_result.json")
        new_result_file = os.path.join(new_path, string, f"{string}_result.json")
        
        # Copy the file
        if os.path.exists(result_file):
            shutil.copy(result_file, new_result_file)
            print(f"Copied {result_file} to {new_result_file}")
        else:
            print(f"File {result_file} does not exist, skipping copying.")
            
def main():
    gw_event_list = ["GW170817", "GW190425", "GW230529"]
    for gw_event in gw_event_list:
        print(f"Processing {gw_event}...")
        copy_gw_event(gw_event)
        
if __name__ == "__main__":
    main()