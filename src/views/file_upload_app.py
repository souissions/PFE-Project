import streamlit as st
import requests

API_URL = "http://localhost:5000"
PROJECT_ID = 1

def upload_and_process_file():
    st.title("🧠 RAG File Uploader")

    st.markdown("Upload a file. It will be automatically chunked, embedded, and stored as vectors.")

    uploaded_file = st.file_uploader(
        "Choose a file (TXT, PDF, DOCX)", 
        type=["txt", "pdf", "docx"],
        help="Max size: 200MB"
    )

    if uploaded_file is not None:
        st.write(f"📄 File selected: `{uploaded_file.name}`")

        if st.button("🚀 Upload and Index"):
            try:
                # Step 1: Upload the file
                files = {
                    "file": (uploaded_file.name, uploaded_file, uploaded_file.type)
                }

                upload_response = requests.post(
                    f"{API_URL}/api/v1/data/upload/{PROJECT_ID}",
                    files=files
                )

                if upload_response.status_code == 200:
                    upload_result = upload_response.json()
                    st.success("✅ File uploaded successfully.")
                    st.write("📦 File ID:", upload_result.get("file_id"))

                    # Step 2: Call /process without file_id
                    st.info("⏳ Processing all project files...")

                    process_payload = {
                        "chunk_size": 200,
                        "overlap_size": 30,
                        "do_reset": 0
                    }

                    process_response = requests.post(
                        f"{API_URL}/api/v1/data/process/{PROJECT_ID}",
                        json=process_payload
                    )

                    if process_response.status_code == 200:
                        process_result = process_response.json()
                        st.success("✅ Project files processed and indexed into vector DB.")

                        # ✅ Proper indentation here
                        with st.expander("📄 Show Processing Details"):
                            st.subheader("🔍 Processing Details")
                            st.write("🧩 Inserted Chunks:", process_result.get("inserted_chunks"))
                            st.write("📂 Processed Files:", process_result.get("processed_files"))
                            st.json(process_result)

                    else:
                        st.error("❌ Processing failed.")
                        st.json(process_response.json())

                else:
                    st.error("❌ Upload failed.")
                    st.json(upload_response.json())

            except Exception as e:
                st.error(f"⚠️ Error: {str(e)}")

# Run the app
upload_and_process_file()
