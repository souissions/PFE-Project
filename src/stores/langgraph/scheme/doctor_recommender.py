import pandas as pd
import numpy as np
import faiss
from sklearn.metrics.pairwise import cosine_similarity
import re
import logging
from typing import Optional, List, Dict, Any

logger = logging.getLogger("uvicorn")

class DoctorRecommender:
    def __init__(self, doctor_df: pd.DataFrame, cases_df: pd.DataFrame, embedding_model):
        """Initialize the doctor recommender with required data and model."""
        self.doctor_df = doctor_df
        self.cases_df = cases_df
        self.embedding_model = embedding_model
        logger.info("👨‍⚕️ Doctor recommender initialized")

    def extract_specialist(self, recommendation_text: Optional[str], specialists_list: List[str]) -> Optional[str]:
        """Extracts specialist category from text using a predefined list."""
        if not recommendation_text or not specialists_list:
            return None
            
        logger.info(f"🔍 Extracting specialist from: '{recommendation_text[:100]}...'")
        recommendation_lower = recommendation_text.lower()
        specialists_list_sorted = sorted(specialists_list, key=len, reverse=True)
        
        for specialist in specialists_list_sorted:
            try:
                pattern = r'\b' + re.escape(specialist.lower()) + r'\b'
                if re.search(pattern, recommendation_lower):
                    logger.info(f"✅ Extracted specialist: {specialist}")
                    return specialist
            except re.error as e:
                logger.error(f"❌ Regex error processing specialist '{specialist}': {e}")
                continue
                
        logger.info("ℹ️ No specific specialist match found")
        return None

    def recommend_doctors(
        self,
        category: Optional[str],
        symptoms: str,
    ) -> Optional[pd.DataFrame]:
        """Finds and recommends doctors based on category and symptoms."""
        logger.info(f"🔍 Finding doctors for category: {category}")
        
        if not self._validate_inputs(category, symptoms):
            return pd.DataFrame()

        try:
            # Filter dataframes by category
            filtered_doctor_df = self._filter_by_category(category)
            filtered_cases_df = self._filter_cases_by_category(category)
            
            if filtered_cases_df.empty:
                logger.warning(f"⚠️ No historical cases found for category: {category}")
                return pd.DataFrame()

            # Generate embeddings and build FAISS index
            embeddings_np, case_details_map = self._prepare_embeddings(filtered_cases_df)
            
            # Search for similar cases
            similar_cases_df = self._find_similar_cases(symptoms, embeddings_np, case_details_map)
            
            if similar_cases_df.empty:
                logger.warning("⚠️ No similar cases found")
                return pd.DataFrame()

            # Aggregate and rank doctors
            recommended_doctors = self._rank_doctors(similar_cases_df, filtered_doctor_df)
            
            # Clean and format final recommendations
            final_recommendation_df = self._clean_recommendations(recommended_doctors)
            
            logger.info(f"✅ Found {len(final_recommendation_df)} doctor recommendations")
            return final_recommendation_df

        except Exception as e:
            logger.error(f"❌ Error in doctor recommendation: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    def _validate_inputs(self, category: Optional[str], symptoms: str) -> bool:
        """Validate input parameters."""
        if not category or not symptoms:
            logger.error("❌ Missing required inputs")
            return False
        if self.doctor_df is None or self.cases_df is None or self.embedding_model is None:
            logger.error("❌ Missing required data or model")
            return False
        return True

    def _filter_by_category(self, category: str) -> pd.DataFrame:
        """Filter doctors by category."""
        return self.doctor_df[self.doctor_df["Specialty"].str.lower() == category.lower()].copy()

    def _filter_cases_by_category(self, category: str) -> pd.DataFrame:
        """Filter cases by category and validate required columns."""
        filtered_cases = self.cases_df[self.cases_df["Specialty"].str.lower() == category.lower()].copy()
        required_cols = ["Symptom Description", "Doctor ID", "Patient Feedback Rating"]
        
        if not all(col in filtered_cases.columns for col in required_cols):
            logger.error("❌ Required columns missing in patient cases data")
            return pd.DataFrame()
            
        filtered_cases.dropna(subset=required_cols, inplace=True)
        return filtered_cases

    def _prepare_embeddings(self, filtered_cases_df: pd.DataFrame) -> tuple:
        """Prepare embeddings and case details for similarity search."""
        symptom_texts = filtered_cases_df["Symptom Description"].tolist()
        logger.info(f"📊 Generating embeddings for {len(symptom_texts)} cases")
        
        embeddings_list = self.embedding_model.embed_documents(symptom_texts)
        embeddings_np = np.array(embeddings_list).astype("float32")
        
        case_details_map = [
            {
                'Doctor ID': filtered_cases_df.iloc[i]["Doctor ID"],
                'Symptom': filtered_cases_df.iloc[i]["Symptom Description"],
                'Rating': filtered_cases_df.iloc[i]["Patient Feedback Rating"]
            }
            for i in range(len(filtered_cases_df))
        ]
        
        return embeddings_np, case_details_map

    def _find_similar_cases(
        self,
        symptoms: str,
        embeddings_np: np.ndarray,
        case_details_map: List[Dict]
    ) -> pd.DataFrame:
        """Find similar cases using FAISS and cosine similarity."""
        logger.info("🔍 Searching for similar cases")
        
        # Build FAISS index
        d = embeddings_np.shape[1]
        temp_index = faiss.IndexFlatL2(d)
        temp_index.add(embeddings_np)
        
        # Search
        query_embedding = np.array([self.embedding_model.embed_query(symptoms)]).astype("float32")
        top_k = min(10, temp_index.ntotal)
        distances, indices = temp_index.search(query_embedding, top_k)
        
        # Process results
        similar_cases_data = []
        for i_idx in indices[0]:
            if i_idx >= 0 and i_idx < len(case_details_map):
                case_info = case_details_map[i_idx]
                case_embedding = embeddings_np[i_idx]
                if query_embedding.shape[1] == case_embedding.shape[0]:
                    sim_score = cosine_similarity(query_embedding, [case_embedding])[0][0]
                    similar_cases_data.append({
                        'Doctor ID': case_info["Doctor ID"],
                        'Symptom': case_info["Symptom"],
                        'Rating': case_info["Rating"],
                        'Similarity Score': round(float(sim_score), 4)
                    })
        
        return pd.DataFrame(similar_cases_data)

    def _rank_doctors(
        self,
        similar_cases_df: pd.DataFrame,
        filtered_doctor_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Rank doctors based on similar cases and ratings."""
        logger.info("📊 Ranking doctors")
        
        # Aggregate scores
        doctor_scores = similar_cases_df.groupby("Doctor ID").agg(
            avg_rating_from_similar=('Rating', 'mean'),
            num_similar_cases=('Doctor ID', 'count'),
            max_similarity_score=('Similarity Score', 'max')
        ).reset_index()
        
        # Merge with doctor profiles
        doctor_scores['Doctor ID'] = doctor_scores['Doctor ID'].astype(str)
        filtered_doctor_df['Doctor ID'] = filtered_doctor_df['Doctor ID'].astype(str)
        
        recommended_doctors = pd.merge(
            doctor_scores,
            filtered_doctor_df,
            on="Doctor ID",
            how="left"
        )
        
        # Sort by relevance metrics
        sort_cols = ["max_similarity_score", "avg_rating_from_similar", "num_similar_cases"]
        sort_cols_present = [col for col in sort_cols if col in recommended_doctors.columns]
        
        if sort_cols_present:
            recommended_doctors = recommended_doctors.sort_values(
                by=sort_cols_present,
                ascending=[False] * len(sort_cols_present)
            )
        
        return recommended_doctors.head(5)

    def _clean_recommendations(self, recommended_doctors: pd.DataFrame) -> pd.DataFrame:
        """Clean and format the final recommendations."""
        logger.info("🧹 Cleaning recommendations")
        
        # Rename columns
        recommended_doctors.rename(columns={
            'avg_rating_from_similar': 'Avg Rating (Similar Cases)',
            'num_similar_cases': 'Similar Cases Found',
            'max_similarity_score': 'Max Similarity Score'
        }, inplace=True)
        
        # Define display columns
        display_cols = [
            "Doctor ID", "Name", "Specialty", "Avg Rating (Similar Cases)",
            "Max Similarity Score", "Similar Cases Found",
            "Years of Experience", "Affiliation"
        ]
        
        # Clean and format each column
        for col in display_cols:
            if col not in recommended_doctors.columns:
                recommended_doctors[col] = None
            if col in recommended_doctors.columns:
                if recommended_doctors[col].isnull().all():
                    recommended_doctors[col] = "" if col in ['Doctor ID', 'Name', 'Specialty', 'Affiliation'] else 0
                else:
                    try:
                        if col in ['Doctor ID', 'Name', 'Specialty', 'Affiliation']:
                            recommended_doctors[col] = recommended_doctors[col].astype(str)
                        elif col in ['Avg Rating (Similar Cases)', 'Max Similarity Score']:
                            recommended_doctors[col] = pd.to_numeric(recommended_doctors[col], errors='coerce').fillna(0.0).round(2)
                        elif col in ['Similar Cases Found', 'Years of Experience']:
                            recommended_doctors[col] = pd.to_numeric(recommended_doctors[col], errors='coerce').fillna(0).astype(int)
                    except Exception as e:
                        logger.error(f"❌ Error formatting column '{col}': {e}")
                        recommended_doctors[col] = "" if col in ['Doctor ID', 'Name', 'Specialty', 'Affiliation'] else 0
        
        return recommended_doctors[display_cols] 