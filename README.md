# Curriculum Learning on Google Research Football Environment
### Setting up GRF and dependencies for Windows
Setting up the environment might be quite the hastle since it requires a lot of dependencies to be run. <br>


1) Install system prereqs <br>
   A. Visual Studio Build Tools <br>
   Download and install Build Tools for Visual Studio 2019 (or 2022) from https://visualstudio.microsoft.com/downloads/.<br>
   During install, select “Desktop development with C++”. <br>
   B. CMake <br>
   Download the Windows installer from https://cmake.org/download/. <br>
   Check “Add CMake to the system PATH” during install. <br>
   Verify in a new terminal: <br>
   cmake --version <br>
   C. Download python 3.9 and make sure to create your .venv with it <br>

2) Install & Configure vcpkg <br>
In your powershell terminal: <br>
cd C:\ <br>
git clone https://github.com/microsoft/vcpkg.git C:\vcpkg <br>
cd C:\vcpkg <br>
.\bootstrap-vcpkg.bat <br>

3) Install SDL2 and Boost (and other deps) in your powershell: <br>
.\vcpkg.exe install sdl2 sdl2-image sdl2-ttf sdl2-gfx boost <br>

Set environment variables for this session in your powershell: <br>
set VCPKG_ROOT=C:\vcpkg <br>
set CMAKE_TOOLCHAIN_FILE=C:\vcpkg\scripts\buildsystems\vcpkg.cmake <br>
(If you want them permanently, add them in System → Environment Variables.) <br>

4) Create a virtual environment <br>

From your project root powershell terminal (or use set environment functionaliy of your IDE): <br>
python -m venv .venv <br>
.\.venv\Scripts\activate <br>
install the requirements from the requirements.txt file <br>

5) Install python dependencies <br>
   pip install --upgrade pip setuptools wheel psutil <br>

6) Build & Install the gfootball Engine <br>
   Load the VS compiler environment in the same powershell terminal: <br>
   cmd /k ""C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64" <br>
   (This drops you into a CMD prompt with the right paths.) <br> 

Back in your project folder (in that CMD prompt): <br>
cd C:\path-to-your-folder\GoogleFootball-C+HRL <br>
pip install --no-cache-dir gfootball <br>

Verify by running: <br>
python -m gfootball.play_game --action_set=full <br>
You should see a window pop up and be able to play. <br>

7) Fix the SDL Downgrade (if you get the Dynamic linking causes SDL downgrade! (compiled with 2.28.4, linked to 2.0.16) issue) <br>
   overwrite the existing SDL2.dll in the GoogleFootball-C+HRL\.venv\Lib\site-packages\gfootball_engine folder with the SDL2.dll file from main folder <br>

try again: <br>
python -m gfootball.play_game --action_set=full

### To train an agent
Go to main.ipynb and run the cells! You will see that the longest cell in the notebook is the training script, it will show you for which scenerio you are training and how far you are in your training. <br>

# Next Steps: <br> 
- Train the agent on all scenerios <br>
- Add a hierarchical design that uses different rewards to optimize for different parts of the game. Some examples: <br>
1) Learn defending if the ball is in your half and the opponent is attacking <br>
2) Learn creating an attack if the ball is in your half and you have the ball <br>
3) Learn to score if the ball is in the opponent's half and you have the ball <br>
4) Learn to press if the ball is in the opponent's half and they have the ball <br>

# TA Feedback
Methodology (34/40): 
<br> - Abstract, introduction and motivation: The authors implemented PPO in the GRF environment. This focuses on using a typical RL algorithm in a widely used environment with curriculum design, which is a suitable course project topic.
<br> - Background and problem formulation: The choice of problem settings is clean and simple, respecting the limited resources. Considering the non-trivial research question and the large number of design choices to try, it's a valid pick for the problem settings.
<br> Related work: well done!
<br> - Contribution and clarity of proposed method: The contribution is satisfactory. The method is clearly stated. This work could be further extended by trying 1-2 more policy-based algorithms on more configurations of the GRF environment
<br> Experiments and coding (18/20): The readme is clearly presented with instructions to run the code easily. The experiments are, in general, carefully designed. The results are well presented. Adding training curves of different curricula in the appendix would be better.
<br> Writing (10/10): Excellent writing.
<br> Presentation (18/20): Overall, great presentation. The video slightly exceeds 4mins. Try to focus on key contributions and findings.
<br> Late penalty: (2^3)% * 90 = 7.2
<br> Well done! Best of luck in your future endeavours!