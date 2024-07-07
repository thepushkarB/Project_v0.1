import gradio as gr
import numpy as np
from joblib import load

# Load the models from files
trained_model = load('trained_model.pkl')
model4 = load('model4.pkl')

# Define dictionaries to map each dropdown option to a specific number
gender_mapping = {"Male": 0, "Female": 1}
backlogs_mapping = {"None": 0, "Less than 5": 1, "Greater than 5": 2}
sleep_mapping = {"Less than 7hr": 0, "7hr": 1, "More than 7hr": 2}
concentration_mapping = {"15min - 30min": 0, "30min - 45min": 1, "More than 45min": 2}
age_mapping = {"15-20": 0, "20-25": 1, "25-30": 2}
academic_level_mapping = {"HIGHER SECONDARY": 0, "UNDER GRADUATE": 1, "POST GRADUATE": 2}
satisfaction_mapping = {"Not at all": 0, "Neutral": 1, "Slightly": 2, "Extremely": 3}
absent_days_mapping = {"Less than 10": 0, "Less than 30": 1, "More than 30": 2}
info_delivery_mapping = {
    "Read/Write": 0,
    "Visual(Pictures / Diagrams)": 1,
    "Auditory": 2,
    "Kinaesthetic (learning by physical activity and hands-on experiences.)": 3
}
study_time_mapping = {"Less than 1 hour": 0, "1-3 hours": 1, "more than 3 hours": 2}
grades_mapping = {"Less than 50%": 0, "50-70%": 1, "70-90%": 2, "Greater than 90%": 3}

# Define dictionaries to map predictions from models to specific text
trained_model_text_mapping = {
    0: "Eisenhower Matrix\n\nList Your Tasks:\n\nBegin by creating a comprehensive list of all the tasks and activities you need to complete. This can include work-related tasks, personal commitments, household chores, and any other responsibilities.\nAssess Urgency and Importance:\nFor each task, evaluate its urgency and importance:\nUrgent: Tasks that require immediate attention or have impending deadlines.\nImportant: Tasks that contribute significantly to your long-term goals, personal growth, or well-being.\nDetermine whether a task is urgent, important, both, or neither.\nPlace Tasks in Quadrants:\nCategorize each task into one of the four quadrants:\nQuadrant 1 (Urgent and Important):\nThese tasks demand immediate action to prevent negative consequences. They are critical and cannot be ignored.\nExamples: Meeting urgent work deadlines, handling emergencies, addressing health issues.\nQuadrant 2 (Important, but Not Urgent):\nThese tasks contribute to your long-term success and well-being. They are essential but don’t have immediate deadlines.\nExamples: Planning, goal setting, self-improvement, relationship-building.\nQuadrant 3 (Urgent, but Not Important):\nThese tasks are urgent but do not significantly impact your long-term goals. Consider delegating or minimizing them.\nExamples: Interruptions, some emails, routine administrative tasks.\nQuadrant 4 (Not Urgent and Not Important):\nThese tasks are neither urgent nor important. They often lead to time-wasting or distractions.\nExamples: Mindless scrolling on social media, trivial activities.\nVisualize your tasks in these quadrants to gain clarity.\nPrioritize Tasks:\nPrioritize your actions based on the quadrants:\nStart with Quadrant 1:\nAddress urgent and important tasks promptly. These require immediate attention.\nMove to Quadrant 2:\nAllocate time for important tasks that contribute to your long-term success. Schedule them to prevent them from becoming urgent.\nDelegate Quadrant 3 Tasks:\nIf possible, delegate tasks that are urgent but not important. Free up your time for more impactful activities.\nMinimize Quadrant 4 Activities:\nAvoid or limit activities that fall into this quadrant. They offer little value.\nCreate a daily or weekly plan based on these priorities.\nSchedule Tasks:\nAllocate specific time slots in your schedule for Quadrants 1 and 2 tasks. Treat them as appointments.\nUse tools like calendars or task management apps to organize your time effectively.\nRegularly Review and Update:\nPeriodically review your task list and priorities. Circumstances change, and new tasks arise.\nAdjust your focus and priorities as needed to maintain balance and productivity.",
    1: "Feynman Technique\n\nChoose a Concept:\nStart by selecting the concept or topic you want to learn or understand better. It could be a theory, principle, formula, or any other subject matter.\nStudy the Concept:\nRead and study the material related to the concept thoroughly. Use textbooks, articles, lectures, or any other resources available to gain a comprehensive understanding of the topic.\nExplain the Concept Simply:\nPretend you’re teaching the concept to someone else who has no prior knowledge of the subject.\nExplain it in simple, easy-to-understand language, avoiding jargon and technical terms.\nThis forces you to break down the concept into its fundamental components and understand it at a deeper level.\nIdentify Knowledge Gaps:\nAs you explain the concept, pay attention to areas where you struggle to explain or where your understanding is unclear.\nThese are potential knowledge gaps that need further study and clarification.\nReview and Simplify:\nReview the concept and your explanation, identifying any areas that are still unclear or overly complicated.\nSimplify your explanation further, using analogies, examples, and everyday language to make it more accessible.\nTeach it to Someone Else:\nOnce you’ve refined your explanation, teach the concept to someone else—a friend, family member, study group, or even an imaginary audience.\nTeaching forces you to solidify your understanding and identify any remaining gaps or misconceptions.\nReview and Repeat:\nAfter teaching the concept, review your explanation and any feedback you received.\nIf there are still areas of confusion, go back and study those parts again.\nRepeat the process until you can confidently and clearly explain the concept.",
    2:"Flowtime Technique\n\nSet Flowtime Blocks:\nDivide your work or study session into flowtime blocks, typically ranging from 25 to 50 minutes each.\nThis duration should be long enough to make progress on tasks but short enough to maintain focus.\nChoose a Task:\nSelect a specific task or activity to focus on during each flowtime block.\nChoose tasks that require deep concentration and can be completed within the allotted time frame.\nEliminate Distractions:\nMinimize distractions and interruptions during your flowtime blocks.\nTurn off notifications, close unnecessary tabs or apps, and create a conducive environment for focused work.\nStart the Timer:\nSet a timer for the duration of the flowtime block.\nUse a timer or a specialized app designed for the Flowtime technique to track your work sessions accurately. add this text t\nWork with Intense Focus:\nDuring the flowtime block, work on the chosen task with intense focus and concentration.\nAvoid multitasking and stay fully engaged in the activity at hand.\nTake Short Breaks:\nAfter completing a flowtime block, take a short break lasting 5 to 10 minutes.\nUse this time to rest, stretch, hydrate, or engage in a brief activity unrelated to work.\nReview Progress:\nAt the end of each flowtime block, take a moment to review your progress on the task.\nReflect on what you’ve accomplished and identify any adjustments or improvements for the next flowtime block.\nRepeat the Cycle:\nRepeat the cycle of alternating between flowtime blocks and short breaks throughout your work or study session.\nAim to complete multiple flowtime blocks, staying in a state of flow for extended periods.\nAdjust as Needed:\nBe flexible and willing to adjust the duration of flowtime blocks or break intervals based on your energy levels and focus.",
    3: "SMART Goals Technique\n\nSpecific (S):\nClearly define your goal. Be precise about what you want to achieve.\nExample: “Increase monthly sales by 15%.”\nMeasurable (M):\nQuantify your goal. Set specific metrics to track progress.\nExample: “Achieve 15% growth within the next quarter.”\nAchievable (A):\nEnsure your goal is realistic and attainable given available resources.\nExample: “Given our current team size, this growth rate is achievable.”\nRelevant (R):\nAlign the goal with your overall objectives and priorities.\nExample: “Sales growth supports our company’s expansion strategy.”\nTime-Bound (T):\nSet a deadline for achieving the goal.\nExample: “Reach the target by the end of Q3.”"
}

model4_text_mapping = {
    0: "Pomodoro Technique\n\nSet a Timer:\nChoose a task you want to work on and set a timer for 25 minutes, known as one “Pomodoro” interval.\nWork on the Task:\nDuring the 25-minute interval, focus solely on the task at hand.\nAvoid distractions and interruptions as much as possible. If any distractions arise, jot them down for later and return to your task.\nTake a Short Break:\nAfter completing the 25 minutes, take a short break, typically 5 minutes.\nUse this time to rest, stretch, or do something unrelated to work to recharge your energy.\nRepeat:\nOnce the break is over, start another Pomodoro session by setting the timer for another 25 minutes and focusing on the task again.\nRepeat this process until you complete four Pomodoros.\nLonger Break:\nAfter completing four Pomodoros (i.e., four 25-minute work intervals), take a longer break, typically 15-30 minutes.\nUse this time to relax, reward yourself, or engage in a different activity before starting another set of Pomodoros.\nTrack Progress:\nKeep track of your completed Pomodoros and breaks using a pen and paper or a digital tool.\nThis helps you monitor your productivity and identify patterns in your work habits.\nAdjust as Needed:\nAdapt the Pomodoro Technique to suit your workflow and preferences.\nYou can adjust the length of the Pomodoro intervals and breaks based on your concentration level and the nature of the task.",
    1: "Time Blocking Technique\n\nIdentify Priorities:\nStart by identifying your most important tasks or priorities for the day or week.\nThese could include work projects, studying, personal tasks, exercise, or relaxation time.\nAllocate Time Blocks:\nAllocate specific blocks of time in your schedule for each task or activity.\nEstimate how much time you’ll need for each task realistically, considering factors like complexity and urgency.\nCreate a Schedule:\nUse a planner, calendar, or scheduling app to create a visual representation of your time blocks for the day or week.\nMake sure to include breaks and transition periods between tasks.\nStick to the Plan:\nFollow your schedule rigorously, focusing on the task at hand during each time block.\nMinimize distractions and avoid multitasking to maximize productivity and effectiveness.\nBe Flexible:\nWhile it’s important to stick to your schedule as much as possible, be flexible and willing to adjust if unexpected events arise or if a task takes longer than expected.\nLearn from these adjustments to improve future time blocking.\nReview and Reflect:\nAt the end of each day or week, review how well you followed your time blocking schedule.\nReflect on what worked well and what didn’t, and make adjustments as needed to optimize your schedule for better productivity and work-life balance.\nPrioritize Self-Care:\nRemember to allocate time blocks for self-care activities such as exercise, relaxation, and spending time with loved ones.\nBalancing work and personal life is crucial for overall well-being and productivity."
}

# Define the predict function
def predict(gender, age, sleep, academic_level, backlogs, concentration, currently_satisfied, absent_days, info_delivery, raise_hands, study_time, grades):
    # Convert each input using the mapping dictionaries
    gender_num = gender_mapping[gender]
    backlogs_num = backlogs_mapping[backlogs]
    sleep_num = sleep_mapping[sleep]
    concentration_num = concentration_mapping[concentration]
    age_num = age_mapping[age]
    academic_level_num = academic_level_mapping[academic_level]
    satisfaction_num = satisfaction_mapping[currently_satisfied]
    absent_days_num = absent_days_mapping[absent_days]
    info_delivery_num = info_delivery_mapping[info_delivery]
    study_time_num = study_time_mapping[study_time]
    grades_num = grades_mapping[grades]
    
    # Convert raise_hands directly as it's already a number
    raise_hands_num = raise_hands
    
    # Combine the input data into an array
    input_data = [
        gender_num,
        age_num,
        sleep_num,
        academic_level_num,
        backlogs_num,
        concentration_num,
        satisfaction_num,
        absent_days_num,
        info_delivery_num,
        raise_hands_num,
        study_time_num,
        grades_num
    ]

    # Convert the list to a numpy array and reshape it for the models
    input_data_reshaped = np.array(input_data).reshape(1, -1)

    # Predict using the models
    y_pred_trained_model = trained_model.predict(input_data_reshaped)
    y_pred_model4 = model4.predict(input_data_reshaped)

    # Retrieve text corresponding to the predictions from the dictionaries
    trained_model_text = trained_model_text_mapping[y_pred_trained_model[0]]
    model4_text = model4_text_mapping[y_pred_model4[0]]

    # Return the predictions
    return trained_model_text, model4_text

# Create the Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Dropdown(["Male", "Female"], label="Gender"),
        gr.Dropdown(["15-20", "20-25", "25-30"], label="Age"),
        gr.Dropdown(["Less than 7hr", "7hr", "More than 7hr"], label="How long do you sleep for on a regular day?"),
        gr.Dropdown(["HIGHER SECONDARY", "UNDER GRADUATE", "POST GRADUATE"], label="Academic Level"),
        gr.Dropdown(["None", "Less than 5", "Greater than 5"], label="Backlogs"),
        gr.Dropdown(["15min - 30min", "30min - 45min", "More than 45min"], label="How long are you able to concentrate on your studies for a continuous period of time?"),
        gr.Dropdown(["Not at all", "Neutral", "Slightly", "Extremely"], label="How much do you enjoy the way you're learning stuff right now?"),
        gr.Dropdown(["Less than 10", "Less than 30", "More than 30"], label="How often would say you were absent in your last semester?"),
        gr.Dropdown([
            "Read/Write",
            "Visual(Pictures / Diagrams)",
            "Auditory",
            "Kinaesthetic (learning by physical activity and hands-on experiences.)"
        ], label="What type of information delivery works best for you?"),
        gr.Slider(0, 5, step=1, label="How often do you raise hands or speak up in class?"),
        gr.Dropdown(["Less than 1 hour", "1-3 hours", "more than 3 hours"], label="How long do you study for on a regular day?"),
        gr.Dropdown(["Less than 50%", "50-70%", "70-90%", "Greater than 90%"], label="How did you do in your most recent important exam, like the final semester exam or boards?")
    ],
    outputs=[
        gr.Textbox(label="Learning strategy"),
        gr.Textbox(label="Time Management Strategies")
    ],
    title="Student Performance Analysis"
)

# Launch the Gradio app
iface.launch(share=True)
