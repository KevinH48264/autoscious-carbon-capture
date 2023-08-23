def initial_prompt():
    return '''
Task:
Design a carbon capture system that is most efficient and with scientific / foundational grounding and reasoning. Conclude with the projected measurement: what is the projected carbon capture operational efficiency of the system in GJ / ton CO2? If you don't have enough details, improve the design so that you can calculate this. Feel free to make assumptions.

Make sure that every detail of the design is written and testable and has a projected energy consumption rate in GJ / ton CO2. Do not give any discretion or choice to the reader. You must propose a method, and the reader must only be responsible for gathering materials and equipment and running in person protocols, not a literature review or choosing materials and methods.

Think step by step and reason yourself to the right decisions to make sure we get it right.
You will first lay out the names of each step of the design that will be necessary, as well as a quick sentence on their purpose, and a rough project energy consumption rate in GJ / ton CO2.

Then you will write the details of each step including ALL materials, equipment, protocols, expected results, and a finer projected energy consumption rate in GJ / ton CO2.

You will start with the first step, then go to the next step, and so on.
Please note that the design should be testable in the short term, and testable. No future technology or materials.

Follow the scientific method and make sure that every step is well reasoned and explained.
Make sure that all steps contain all materials, equipment, protocols, expected results, etc. Makes sure that details in different steps are compatible with each other.
Ensure to write all details, if you are unsure, write a plausible answer, but note "(uncertain)".

Before you finish, double check that all parts of the design are present in the steps and the final total projected energy consumption rate is reasonable.

Some potentially useful and relevant information (optional to use):
Carbon capture is one of the foremost methods for curtailing greenhouse gas emissions. Incumbent technologies are inherently inefficient due to thermal energy losses, large footprint, or degradation of sorbent material. We report a solid-state faradaic electro-swing reactive adsorption system comprising an electrochemical cell that exploits the reductive addition of CO2 to quinones for carbon capture. The reported device is compact and flexible, obviates the need for ancillary equipment, and eliminates the parasitic energy losses by using electrochemically activated redox carriers. An electrochemical cell with a polyanthraquinone–carbon nanotube composite negative electrode captures CO2 upon charging via the carboxylation of reduced quinones, and releases CO2 upon discharge. The cell architecture maximizes the surface area exposed to gas, allowing for ease of stacking of the cells in a parallel passage contactor bed. We demonstrate the capture of CO2 both in a sealed chamber and in an adsorption bed from inlet streams of CO2 concentrations as low as 0.6% (6000 ppm) and up to 10%, at a constant CO2 capacity with a faradaic efficiency of >90%, and a work of 40–90 kJ per mole of CO2 captured, with great durability of electrochemical cells showing <30% loss of capacity after 7000 cylces.
'''

def criticize_prompt():
    return '''
Find problems with this design.
'''

def improve_prompt():
    return '''
Based on this, the improved design to solve the task is as follows.
'''