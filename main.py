from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches
import uvicorn

app = FastAPI(title="Course Recommendation API", version="2.0")

# Global variables
data = None
similarity_matrix = None


# ---------- STARTUP EVENT ----------
@app.on_event("startup")
def load_data():
    """
    Load dataset and prepare similarity matrix when FastAPI starts.
    """
    global data, similarity_matrix

    try:
        # Load dataset (make sure courses.csv is in same folder)
        data = pd.read_csv("courses.csv")
        data.fillna("", inplace=True)

        # Combine text columns for semantic similarity
        data["combined"] = (
            data["Course Name"].astype(str)
            + " " + data["Course Description"].astype(str)
            + " " + data["Skills"].astype(str)
        )

        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(data["combined"])

        # Compute similarity matrix
        similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

        print(f"Dataset loaded successfully with {len(data)} courses.")

    except Exception as e:
        print(f"Error loading dataset: {e}")


# ---------- ROOT ENDPOINT ----------
@app.get("/")
def home():
    return {"message": "Welcome to the Course Recommendation API 🚀"}


# ---------- LIST COURSES ----------
@app.get("/courses/")
def get_courses(limit: int = Query(10, description="Number of courses to return")):
    """
    Return a list of available course names.
    """
    if data is None:
        return JSONResponse(status_code=400, content={"error": "Dataset not loaded."})

    courses = data["Course Name"].head(limit).tolist()
    return {"total_courses": len(data), "sample_courses": courses}


# ---------- RECOMMEND COURSES (FUZZY MATCH ENABLED) ----------
@app.get("/recommend/")
def recommend_courses(course_name: str = Query(..., description="Enter course name or partial name")):
    """
    Recommend similar courses based on the given course name (supports fuzzy matching).
    """
    if data is None or similarity_matrix is None:
        return JSONResponse(status_code=400, content={"error": "Please upload or load a dataset first."})

    course_name = course_name.strip().lower()
    all_courses = data["Course Name"].str.lower().tolist()

    # Fuzzy match to find closest course name
    match = get_close_matches(course_name, all_courses, n=1, cutoff=0.5)

    if not match:
        return JSONResponse(
            status_code=404,
            content={"error": f"No close match found for '{course_name}'."}
        )

    matched_course = match[0]
    idx = data[data["Course Name"].str.lower() == matched_course].index[0]

    # Get similarity scores
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Top 5 similar courses (excluding itself)
    top_courses = []
    for i, score in sim_scores[1:6]:
        top_courses.append({
            "Course Name": data.iloc[i]["Course Name"],
            "Similarity Score": round(float(score), 3),
            "Difficulty Level": data.iloc[i].get("Difficulty Level", ""),
            "Course Description": data.iloc[i]["Course Description"][:150] + "..."
        })

    return {
        "matched_input": data.iloc[idx]["Course Name"],
        "recommendations_found": len(top_courses),
        "recommended_courses": top_courses
    }


# ---------- SEARCH COURSES ----------
@app.get("/search/")
def search_courses(keyword: str = Query(..., description="Keyword to search in course name, description, or skills")):
    """
    Search for courses containing the given keyword.
    Example: /search/?keyword=python
    """
    if data is None:
        return JSONResponse(status_code=400, content={"error": "Dataset not loaded."})

    keyword = keyword.lower().strip()

    # Filter rows by keyword in text columns
    mask = (
        data["Course Name"].str.lower().str.contains(keyword, na=False)
        | data["Course Description"].str.lower().str.contains(keyword, na=False)
        | data["Skills"].str.lower().str.contains(keyword, na=False)
    )

    results = data[mask][["Course Name", "Difficulty Level", "Course Description"]].head(10)

    if results.empty:
        return {"message": f"No courses found for keyword '{keyword}'."}

    courses = results.to_dict(orient="records")
    return {"keyword": keyword, "matches_found": len(courses), "courses": courses}


# ---------- RUN SERVER ----------
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)