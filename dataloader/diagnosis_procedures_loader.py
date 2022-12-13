import sqlite3 as sql
from typing import List
import pandas as pd

class DiagnosisProceduresLoader:
    """Connects to insurance claims database to return insights about doctor-specific diagnoses and procedure referrals."""
    def __init__(self, database_path: str, diagnosis_codes: List[str], procedure_codes: List[str], excluded_diagnoses: str):
        """Create a DiagnosisProceduresLoader object and connect to the database.
        
        Args:
            database_path (str): Path to the database file.
            diagnosis_codes (list(str)): ICD-9 codes specifying diagnoses of interest.
            procedure_codes (list(str)): CPT codes specifying procedures of interest.
            excluded_diagnoses (str): SQL varchar pattern of diagnoses to exclude from query (ex. '152.%' for malignant tumors)
        """
        self.database_path = database_path
        self.connection = sql.connect(database_path)
        self.cur = self.connection.cursor()
        self.placeholder = '?'
        self.diagnosis_codes = diagnosis_codes
        self.procedure_codes = procedure_codes
        self.excluded_diagnoses = excluded_diagnoses

    def get_doctor_diagnosis_procedure_data(self) -> pd.DataFrame:
        """Return a dataframe containing doctor-specific information for number of patients diagnosed and/or operated on as specified.
        
        Columns of returned dataframe:
            doctor_id (str): Unique identifier for each qualifying doctor the database.
            num_diagnosed (int): Number of patients diagnosed by each doctor as specified by `self.diagnosis_codes`
            num_operated_on (int): Number of patients who were referred by each doctor to a procedure as specified by `self.procedure_codes`
        """
        # Create diagnosis and procedure tables to aggregate data from.
        self._create_diagnosis_table()
        self._create_procedure_table()
        
        # Run data aggregations.
        query_to_count_diagnosed_patients_operated_on = (
        """
        SELECT diagnosis.doctor_id, COUNT(DISTINCT diagnosis.patient_id) as num_operated_on
        FROM diagnosis_table as diagnosis
        INNER JOIN procedures_table as procedures
            ON diagnosis.encounter_key == procedures.encounter_key

        GROUP BY diagnosis.doctor_id
        ORDER BY num_operated_on DESC;
        """
        )
        result = self.cur.execute(query_to_count_diagnosed_patients_operated_on)
        diagnosed_and_operated_on_df = pd.DataFrame(result.fetchall())
        diagnosed_and_operated_on_df.columns = [ x[0] for x in result.description ]
        diagnosed_and_operated_on_df.set_index("doctor_id", inplace=True)

        query_to_count_total_diagnosed_patients = (
        """
        SELECT doctor_id, COUNT(DISTINCT patient_id) as num_diagnosed
        FROM diagnosis_table
        GROUP BY doctor_id
        ORDER BY num_diagnosed DESC;
        """
        )
        result = self.cur.execute(query_to_count_total_diagnosed_patients)
        diagnosed_df = pd.DataFrame(result.fetchall())
        diagnosed_df.columns = [ x[0] for x in result.description ]
        diagnosed_df.set_index("doctor_id", inplace=True)

        combined_df = diagnosed_df.join(diagnosed_and_operated_on_df, on="doctor_id")
        combined_df["num_operated_on"].fillna(0, inplace=True)

        return combined_df

    def _create_diagnosis_table(self) -> None:
        """Create filtered table from database containing only diagnoses of interest."""

        # To avoid injection attacks, we should only insert the placeholder character into a query
        placeholders = ', '.join(self.placeholder for _ in self.diagnosis_codes)
        num_columns_to_search = 26
        substitution = self.diagnosis_codes*num_columns_to_search + [self.excluded_diagnoses]*num_columns_to_search

        query = (
            f"""
            CREATE TABLE IF NOT EXISTS diagnosis_table AS
            SELECT encounter_key, doctor_id, patient_id
            FROM medical_headers
            WHERE (

                (DA IN ({placeholders}) OR
                D1 IN ({placeholders}) OR
                D2 IN ({placeholders}) OR
                D3 IN ({placeholders}) OR
                D4 IN ({placeholders}) OR
                D5 IN ({placeholders}) OR
                D6 IN ({placeholders}) OR
                D7 IN ({placeholders}) OR
                D8 IN ({placeholders}) OR
                D9 IN ({placeholders}) OR
                D10 IN ({placeholders}) OR
                D11 IN ({placeholders}) OR
                D12 IN ({placeholders}) OR
                D13 IN ({placeholders}) OR
                D14 IN ({placeholders}) OR
                D15 IN ({placeholders}) OR
                D16 IN ({placeholders}) OR
                D17 IN ({placeholders}) OR
                D18 IN ({placeholders}) OR
                D19 IN ({placeholders}) OR
                D20 IN ({placeholders}) OR
                D21 IN ({placeholders}) OR
                D22 IN ({placeholders}) OR
                D23 IN ({placeholders}) OR
                D24 IN ({placeholders}) OR
                D25 IN ({placeholders}))
                
                AND DA NOT LIKE (?)
                AND D1 NOT LIKE (?)
                AND D2 NOT LIKE (?)
                AND D3 NOT LIKE (?)
                AND D4 NOT LIKE (?)
                AND D5 NOT LIKE (?)
                AND D6 NOT LIKE (?)
                AND D7 NOT LIKE (?)
                AND D8 NOT LIKE (?)
                AND D9 NOT LIKE (?)
                AND D10 NOT LIKE (?)
                AND D11 NOT LIKE (?)
                AND D12 NOT LIKE (?)
                AND D13 NOT LIKE (?)
                AND D14 NOT LIKE (?)
                AND D15 NOT LIKE (?)
                AND D16 NOT LIKE (?)
                AND D17 NOT LIKE (?)
                AND D18 NOT LIKE (?)
                AND D19 NOT LIKE (?)
                AND D20 NOT LIKE (?)
                AND D21 NOT LIKE (?)
                AND D22 NOT LIKE (?)
                AND D23 NOT LIKE (?)
                AND D24 NOT LIKE (?)
                AND D25 NOT LIKE (?)
            );
            """
        )
        self.cur.execute(query, substitution)

    def _create_procedure_table(self):
        """Create a filtered table in the database containing only rows corresponding to procedures of interest."""
        
        # To avoid injection attacks, we should only insert the placeholder character into a query
        placeholders = ', '.join(self.placeholder for _ in self.procedure_codes)
        query = (
            f"""
            CREATE TABLE IF NOT EXISTS procedures_table AS
            SELECT encounter_key, procedure
            FROM medical_service_lines
            WHERE procedure IN ({placeholders});
            """
        )
        self.cur.execute(query, self.procedure_codes)

    def __enter__(self):
        """Allows dataloader to be used in a context manager."""
        return self

    def __exit__(self, ext_type, exc_value, traceback):
        """Handles exceptions within context manager for dataloader."""
        self.cur.close()
        if isinstance(exc_value, Exception):
            self.connection.rollback()
        else:
            self.connection.commit()
        self.connection.close()
