import sys
import panda_py

if __name__ == '__main__':

  if len(sys.argv) < 4:
    print(f'Usage: python {sys.argv[0]} <robot-hostname> <fci-username> <fci-password>')
    hostname = "10.0.0.2"
    username = "erol@metu.edu.tr"
    password = "PandaPandaPanda"
  else:
    hostname = sys.argv[1]
    username = sys.argv[2]
    password = sys.argv[3]
  # Open desk in script, can also be done on a browser
  desk = panda_py.Desk(hostname, username, password)
  desk.lock()

  print("Panda locked.")