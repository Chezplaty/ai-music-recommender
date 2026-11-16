# 🎵 Music Recommender Simulation

## Project Summary

In this project you will build and explain a small music recommender system.

Your goal is to:

- Represent songs and a user "taste profile" as data
- Design a scoring rule that turns that data into recommendations
- Evaluate what your system gets right and wrong
- Reflect on how this mirrors real world AI recommenders

Replace this paragraph with your own summary of what your version does.

---

## How The System Works

Real-world recommendations work by finding patterns in a user's own behavior and then accumulating a bunch of data from millions of users to then find universal patterns amongst them. This allows the system to give off recommendations that can be very similar based off of just features like genre or energy, while also allowing for a little bit of mix up here and there from the data of other users.

Since this system will be purely off content-based filtering, it will prioritize key features such as genre, energy, mood, valence and acousticness.

Song objects will have all of these attributes.

UserProfile objects will have a favorite genre and mood, a target energy, whether or not they like acoustic songs, and target valence which indicates how positive/happy they want songs to feel.

A score will then be calculated based off how close all the features are to the user's preference with more weight being on genre and energy.

The score works like this:
score = (
    0.30 * genre_sim
    + 0.15 * mood_sim
    + 0.35 * energy_sim
    + 0.10 * valence_sim
    + 0.10 * acoustic_sim
)

The genre and mood similarities will either be 1 or 0 depending if it is one of the user's favorties. Then an energy, valence, and acoustic similarity score will be calculated by subtracting the two scores.

If songs do not exactly match the genre, the max similarity a song can be is 0.7, but this can still recommended to a user so I think the scoring is pretty fair since only two categories are all or nothing, while the rest is truly calculated based off similarity.
---

## Getting Started

### Setup

1. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Mac or Linux
   .venv\Scripts\activate         # Windows

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Add your Gemini key: copy `.env.example` to `.env` and fill in `GEMINI_API_KEY`.

4. Run the web app:

```bash
streamlit run app.py
```

Create an account, set up your taste profile, then describe what you want to hear.
"Quick mix from my profile" with the AI critic switched off works without Gemini.

Or use the command line: `python src/main.py "chill r&b for studying"`
(`--mode agent|agent-rules|rag|baseline`).

### Running Tests

Run the starter tests with:

```bash
pytest
```

You can add more tests in `tests/test_recommender.py`.

---

## Sample Recommendation Output

Paste a sample of your recommender's output here as a text block so a reader can see what it produces:

=== Chill lofi listener ===

Top recommendations:

Midnight Coding - Score: 0.95
-----------------------------
Because: Matches favorite genre 'lofi'
Matches favorite mood 'chill'
Energy similarity 0.97 (song 0.42, target 0.45)
Valence similarity 0.94 (song 0.56, target 0.5)
Acousticness similarity 0.71 (song 0.71, target 1.0)

Library Rain - Score: 0.94
--------------------------
Because: Matches favorite genre 'lofi'
Matches favorite mood 'chill'
Energy similarity 0.90 (song 0.35, target 0.45)
Valence similarity 0.90 (song 0.6, target 0.5)
Acousticness similarity 0.86 (song 0.86, target 1.0)

Focus Flow - Score: 0.80
------------------------
Because: Matches favorite genre 'lofi'
Different mood ('focused' vs favorites ['chill'])
Energy similarity 0.95 (song 0.4, target 0.45)
Valence similarity 0.91 (song 0.59, target 0.5)
Acousticness similarity 0.78 (song 0.78, target 1.0)

Spacewalk Thoughts - Score: 0.62
--------------------------------
Because: Different genre ('ambient' vs favorites ['lofi'])
Matches favorite mood 'chill'
Energy similarity 0.83 (song 0.28, target 0.45)
Valence similarity 0.85 (song 0.65, target 0.5)
Acousticness similarity 0.92 (song 0.92, target 1.0)

**Screenshot or video** *(optional)*: <!-- Insert a screenshot or demo video link here -->

---

## Experiments You Tried

Use this section to document the experiments you ran. For example:

- What happened when you changed the weight on genre from 2.0 to 0.5
- What happened when you added tempo or valence to the score
- How did your system behave for different types of users

---

## Limitations and Risks

Summarize some limitations of your recommender.

Examples:

- It only works on a tiny catalog
- It does not understand lyrics or language
- It might over favor one genre or mood

You will go deeper on this in your model card.

---

## Reflection

Read and complete `model_card.md`:

[**Model Card**](model_card.md)

Write 1 to 2 paragraphs here about what you learned:

- about how recommenders turn data into predictions
- about where bias or unfairness could show up in systems like this



