"""Static ASU pages: one source each, indexed from the markdown Firecrawl returns."""

from __future__ import annotations

from scraper.core.types import Source

#: Hosts outside asu.edu that a page may live on: student-run sites and the public IT help desk.
OTHER_HOSTS = frozenset(
    {"asu.my.salesforce-sites.com", "www.asuusgt.org", "www.pitchforkpantry.org"}
)


def _page(key: str, url: str, category: str, every: int = 168) -> Source:
    return Source(key=key, url=url, category=category, fetch_every_hours=every)


PAGES: tuple[Source, ...] = (
    # calendar
    _page(
        "registrar_academic_calendar",
        "https://registrar.asu.edu/academic-calendar",
        "calendar",
        72,
    ),
    _page(
        "registrar_final_exam_schedule",
        "https://registrar.asu.edu/final-exam-schedule",
        "calendar",
        72,
    ),
    # registrar
    _page("registrar_drop_add", "https://registrar.asu.edu/drop-add", "registrar"),
    _page("registrar_withdrawal", "https://registrar.asu.edu/forms/withdrawal", "registrar"),
    _page(
        "registrar_late_registration",
        "https://registrar.asu.edu/late-registration",
        "registrar",
    ),
    _page("registrar_grades", "https://registrar.asu.edu/grades", "registrar"),
    _page("registrar_transcripts", "https://registrar.asu.edu/transcripts", "registrar"),
    _page(
        "registrar_enrollment_verification",
        "https://registrar.asu.edu/enrollment-verification",
        "registrar",
    ),
    _page("registrar_residency_navigator", "https://registrar.asu.edu/residency", "registrar"),
    _page(
        "registrar_residency_requirements",
        "https://registrar.asu.edu/residency-requirements",
        "registrar",
    ),
    _page("registrar_leave_of_absence", "https://registrar.asu.edu/leave-absence", "registrar"),
    _page("registrar_faqs", "https://registrar.asu.edu/faqs", "registrar"),
    # graduation
    _page("registrar_graduation_apply", "https://registrar.asu.edu/graduation-apply", "graduation"),
    _page(
        "registrar_graduation_ceremonies",
        "https://registrar.asu.edu/graduation-ceremonies",
        "graduation",
    ),
    _page("registrar_diploma", "https://registrar.asu.edu/diploma", "graduation"),
    _page(
        "graduation_undergrad_commencement",
        "https://graduation.asu.edu/undergraduate-commencement",
        "graduation",
        72,
    ),
    _page(
        "graduation_future_dates",
        "https://graduation.asu.edu/ceremonies/futuredates",
        "graduation",
    ),
    # tuition
    _page(
        "tuition_payment_deadlines",
        "https://tuition.asu.edu/billing-finances/deadlines",
        "tuition",
        72,
    ),
    _page(
        "tuition_refund_policy",
        "https://tuition.asu.edu/billing-finances/tuition-refund",
        "tuition",
    ),
    _page("tuition_billing", "https://tuition.asu.edu/billing-finances", "tuition"),
    _page(
        "tuition_payment_plan",
        "https://tuition.asu.edu/billing-finances/payment-plan",
        "tuition",
    ),
    _page("tuition_costs_fees", "https://tuition.asu.edu/cost/tuition-fees", "tuition"),
    _page(
        "tuition_fee_descriptions",
        "https://tuition.asu.edu/cost-calculator/tuition-fees/fee-descriptions",
        "tuition",
    ),
    # financial_aid
    _page("tuition_cost_of_attendance", "https://tuition.asu.edu/cost", "financial_aid"),
    _page("finaid_fafsa", "https://tuition.asu.edu/financial-aid/fafsa", "financial_aid"),
    _page("finaid_grants", "https://tuition.asu.edu/financial-aid/grants", "financial_aid"),
    _page("finaid_loans", "https://tuition.asu.edu/financial-aid/loans", "financial_aid"),
    _page(
        "finaid_maintaining_aid",
        "https://tuition.asu.edu/financial-aid/maintaining-aid",
        "financial_aid",
    ),
    _page("finaid_sap", "https://tuition.asu.edu/satisfactory-academic-progress", "financial_aid"),
    _page(
        "finaid_disbursement",
        "https://tuition.asu.edu/financial-aid/disbursement",
        "financial_aid",
    ),
    _page(
        "finaid_work_study",
        "https://studentemployment.asu.edu/students/preparing-student-employment/federal-work-study",
        "financial_aid",
    ),
    # advising
    _page("advising_find_advisor", "https://students.asu.edu/advising", "advising"),
    # tutoring
    _page("tutoring_subject_centers", "https://tutoring.asu.edu/tutoring", "tutoring", 72),
    _page("tutoring_writing_centers", "https://tutoring.asu.edu/writing-centers", "tutoring", 72),
    _page(
        "tutoring_graduate_writing",
        "https://tutoring.asu.edu/graduate-writing-centers",
        "tutoring",
        72,
    ),
    # academics
    _page("degrees_bachelors_landing", "https://degrees.asu.edu/bachelors", "academics"),
    _page(
        "admission_transfer_credits",
        "https://admission.asu.edu/apply/transfer/transferring-credits",
        "academics",
    ),
    # honors
    _page(
        "barrett_lower_division",
        "https://students.barretthonors.asu.edu/academics/lower-division-curriculum",
        "honors",
    ),
    _page(
        "barrett_honors_contracts",
        "https://students.barretthonors.asu.edu/academics/honors-enrichment-contracts",
        "honors",
        72,
    ),
    _page(
        "barrett_honors_thesis",
        "https://students.barretthonors.asu.edu/academics/honors-thesis",
        "honors",
    ),
    # graduate
    _page(
        "gradcollege_graduation_deadlines",
        "https://graduate.asu.edu/current-students/policies-forms-and-deadlines/graduation-deadlines",
        "graduate",
        72,
    ),
    _page(
        "gradcollege_calendar",
        "https://graduate.asu.edu/graduate-college-calendar",
        "graduate",
        72,
    ),
    _page(
        "gradcollege_format_manual",
        "https://graduate.asu.edu/current-students/completing-your-degree/formatting-your-thesis-or-dissertation/asu-graduate-college",
        "graduate",
    ),
    _page(
        "gradcollege_ra_ta_conditions",
        "https://graduate.asu.edu/current-students/funding-opportunities/graduate-appointments-and-assistantships/policies-and-procedures/ta",
        "graduate",
    ),
    # international
    _page(
        "issc_full_time_enrollment",
        "https://issc.asu.edu/f-1j-1-students/maintaining-full-time-enrollment",
        "international",
        72,
    ),
    _page(
        "issc_maintaining_status",
        "https://issc.asu.edu/f-1j-1-students/maintaining-status",
        "international",
        72,
    ),
    _page(
        "issc_i20",
        "https://issc.asu.edu/f-1j-1-students/understanding-your-i-20",
        "international",
    ),
    _page("issc_travel", "https://issc.asu.edu/f-1j-1-students/traveling", "international", 72),
    _page(
        "issc_cpt",
        "https://issc.asu.edu/f-1j-1-students/employment/f1-cpt",
        "international",
        72,
    ),
    _page(
        "issc_post_opt",
        "https://issc.asu.edu/f-1j-1-students/employment/off-campus-after",
        "international",
        72,
    ),
    _page(
        "issc_stem_opt",
        "https://issc.asu.edu/f-1j-1-students/employment/stem",
        "international",
        72,
    ),
    _page(
        "issc_faq",
        "https://issc.asu.edu/f-1j-1-students/most-commonly-asked-questions",
        "international",
    ),
    # online
    _page("asuonline_student_services", "https://asuonline.asu.edu/students/services/", "online"),
    _page(
        "asuonline_success_coach",
        "https://currentstudent.asuonline.asu.edu/success-coach/",
        "online",
    ),
    _page(
        "asuonline_transfer_requirements",
        "https://asuonline.asu.edu/admission/transfer/apply/",
        "online",
    ),
    _page("asuonline_first_year", "https://asuonline.asu.edu/admission/first-year/", "online"),
    _page("asuonline_faq", "https://asuonline.asu.edu/about-us/faq/", "online"),
    _page("starbucks_college_plan", "https://starbucks.asu.edu/", "online"),
    # policy
    _page(
        "provost_academic_integrity_policy",
        "https://provost.asu.edu/academic-integrity/policy",
        "policy",
    ),
    _page(
        "provost_academic_integrity_students",
        "https://provost.asu.edu/academic-integrity/resources/students",
        "policy",
    ),
    _page(
        "code_of_conduct_procedures",
        "https://eoss.asu.edu/sites/g/files/litvpz141/files/Student_Code_of_Conduct_Procedures.pdf",
        "policy",
    ),
    # veterans
    _page("veterans_checklist", "https://veterans.asu.edu/veteran-checklist", "veterans"),
    _page("veterans_benefit_types", "https://veterans.asu.edu/va-benefit-types", "veterans"),
    _page("veterans_tuition_assistance", "https://veterans.asu.edu/tuition-assistance", "veterans"),
    # admissions
    _page(
        "admission_first_year_requirements",
        "https://admission.asu.edu/apply/first-year/admission",
        "admissions",
    ),
    _page(
        "admission_first_year_overview",
        "https://admission.asu.edu/apply/first-year",
        "admissions",
        72,
    ),
    _page(
        "admission_first_year_admitted",
        "https://admission.asu.edu/apply/first-year/admitted",
        "admissions",
        72,
    ),
    _page(
        "admission_transfer_requirements",
        "https://admission.asu.edu/apply/transfer/admission",
        "admissions",
    ),
    _page(
        "admission_international_first_year",
        "https://admission.asu.edu/apply/international/first-year",
        "admissions",
    ),
    _page("admission_faqs", "https://admission.asu.edu/contact/faqs", "admissions"),
    _page(
        "admission_resident_tuition_first_year",
        "https://admission.asu.edu/cost-aid/resident-first-year",
        "admissions",
    ),
    _page("changing_majors", "https://changingmajors.asu.edu/", "admissions"),
    # orientation
    _page(
        "orientation_new_student",
        "https://eoss.asu.edu/orientation/new-student",
        "orientation",
        72,
    ),
    _page(
        "orientation_new_student_experience",
        "https://eoss.asu.edu/orientation",
        "orientation",
        72,
    ),
    _page("sun_devil_welcome", "https://eoss.asu.edu/welcome", "orientation", 72),
    # engineering
    _page("fulton_advising", "https://students.engineering.asu.edu/advising/", "engineering"),
    _page(
        "fulton_tutoring",
        "https://students.engineering.asu.edu/pulse/tutoring/",
        "engineering",
        72,
    ),
    _page(
        "fulton_student_orgs",
        "https://students.engineering.asu.edu/organizations/",
        "engineering",
    ),
    _page(
        "fulton_undergrad_faqs",
        "https://engineering.asu.edu/undergraduate-faqs/",
        "engineering",
    ),
    _page(
        "gcsp_about",
        "https://gcsp.engineering.asu.edu/grand-challenge-scholars-program/",
        "engineering",
    ),
    _page("epics", "https://epics.engineering.asu.edu/", "engineering"),
    _page(
        "scai_first_year_transfer",
        "https://scai.engineering.asu.edu/first-year-and-transfer-students/",
        "engineering",
    ),
    _page(
        "scai_cs_bs_degree_requirements",
        "https://scai.engineering.asu.edu/computer-science-bs/degree-requirements/",
        "engineering",
    ),
    _page(
        "scai_undergrad_advising",
        "https://scai.engineering.asu.edu/undergraduate-advising/",
        "engineering",
    ),
    _page(
        "scai_advising_faq",
        "https://scai.engineering.asu.edu/frequently-asked-questions/",
        "engineering",
    ),
    # business
    _page(
        "wpcarey_undergrad_advising",
        "https://wpcarey.asu.edu/undergraduate-degrees/advising",
        "business",
    ),
    _page(
        "wpcarey_current_students",
        "https://wpcarey.asu.edu/undergraduate/current-students",
        "business",
    ),
    _page("wpcarey_career", "https://wpcarey.asu.edu/undergraduate/career", "business"),
    _page(
        "wpcarey_direct_admission",
        "https://wpcarey.asu.edu/undergraduate/direct-admission",
        "business",
    ),
    # research
    _page("fulton_furi", "https://students.engineering.asu.edu/furi/", "research", 72),
    _page(
        "fulton_undergrad_research",
        "https://students.engineering.asu.edu/undergraduate/research/",
        "research",
    ),
    _page(
        "fulton_more",
        "https://students.engineering.asu.edu/graduate/research/more/",
        "research",
    ),
    _page(
        "barrett_thesis",
        "https://students.barretthonors.asu.edu/academics/thesis-creative-project",
        "research",
    ),
    _page("provost_uresearch", "https://provost.asu.edu/uresearch", "research"),
    _page("ours_online_research", "https://ours.thecollege.asu.edu/", "research"),
    _page(
        "sols_research_opportunities",
        "https://sols.asu.edu/student-life/open-research-opportunities",
        "research",
    ),
    # study_abroad
    _page("goglobal_home", "https://goglobal.asu.edu/", "study_abroad"),
    _page("goglobal_deadlines", "https://goglobal.asu.edu/important-deadlines", "study_abroad", 72),
    _page("goglobal_faqs", "https://goglobal.asu.edu/students/faqs", "study_abroad"),
    _page(
        "goglobal_semester_costs",
        "https://goglobal.asu.edu/students/comparing-semester-program-costs",
        "study_abroad",
    ),
    _page(
        "goglobal_scholarships",
        "https://goglobal.asu.edu/students/scholarships-grants",
        "study_abroad",
    ),
    # internships
    _page("fulton_career_center", "https://career.engineering.asu.edu/", "internships", 72),
    _page(
        "scai_internship_credit",
        "https://scai.engineering.asu.edu/uginternships/",
        "internships",
    ),
    _page("wpcarey_coop", "https://wpcarey.asu.edu/co-op", "internships"),
    _page(
        "career_internship_resources",
        "https://career.eoss.asu.edu/internship-resources/",
        "internships",
    ),
    # career
    _page(
        "career_fairs",
        "https://career.eoss.asu.edu/channels/career-internship-fairs/",
        "career",
        72,
    ),
    _page("career_advising", "https://career.eoss.asu.edu/channels/career-advising/", "career", 72),
    # ai
    _page("ai_tools", "https://ai.asu.edu/ai-tools", "ai", 72),
    _page("ai_chatgpt_edu", "https://ai.asu.edu/ai-tools/chatgpt-edu", "ai", 72),
    _page("ai_asu_home", "https://ai.asu.edu/", "ai"),
    _page("ai_technical_foundation", "https://ai.asu.edu/technical-foundation", "ai"),
    _page("ai_createai_builder", "https://ai.asu.edu/technical-foundation/createai-builder", "ai"),
    _page("ai_asu_programs", "https://ai.asu.edu/asu-programs", "ai"),
    _page(
        "ai_asu_openai_expansion",
        "https://tech.asu.edu/features/asu-and-openai-expand-collaboration-scaling-ai",
        "ai",
    ),
    _page(
        "ai_acceleration_launch",
        "https://tech.asu.edu/features/ASU-launches-AI-acceleration",
        "ai",
    ),
    _page("provost_generative_ai", "https://provost.asu.edu/generative-ai", "ai"),
    # entrepreneurship
    _page("edson_ei_home", "https://entrepreneurship.asu.edu/", "entrepreneurship"),
    _page(
        "venture_devils",
        "https://entrepreneurship.asu.edu/programs/venture-devils/",
        "entrepreneurship",
    ),
    _page(
        "venture_devils_demo_day",
        "https://entrepreneurship.asu.edu/funding-resources/venture-devils-demo-day/",
        "entrepreneurship",
        72,
    ),
    _page(
        "edson_funding",
        "https://entrepreneurship.asu.edu/funding-resources/",
        "entrepreneurship",
    ),
    _page("edson_spaces", "https://entrepreneurship.asu.edu/spaces/", "entrepreneurship"),
    _page("changemaker_home", "https://changemaker.asu.edu/", "entrepreneurship"),
    # programs
    _page(
        "tuition_special_assistance_az_residents",
        "https://tuition.asu.edu/special-financial-assistance-programs-arizona-residents",
        "programs",
    ),
    _page(
        "credit_for_service_learning",
        "https://universitycollege.asu.edu/students/experiential-learning/credit-service-learning",
        "programs",
    ),
    _page("trio_sss", "https://eoss.asu.edu/trio/student-support-services", "programs"),
    _page(
        "trio_sss_eligibility",
        "https://eoss.asu.edu/trio/student-support-services/eligibility-and-admission",
        "programs",
    ),
    _page("first_gen_asu", "https://yourfuture.asu.edu/firstgeneration", "programs"),
    # housing
    _page("housing_move_in_rates", "https://housing.asu.edu/move-in", "housing", 72),
    _page("housing_faqs", "https://housing.asu.edu/faqs", "housing"),
    _page(
        "housing_continuing_selection",
        "https://housing.asu.edu/current-resident-housing/housing-selection",
        "housing",
        72,
    ),
    _page(
        "housing_winter_summer",
        "https://housing.asu.edu/winter-and-summer-housing",
        "housing",
        72,
    ),
    _page("housing_resources", "https://housing.asu.edu/resources", "housing"),
    _page(
        "housing_policies",
        "https://housing.asu.edu/housing-resources/housing-policies-and-procedures",
        "housing",
    ),
    _page(
        "housing_how_to_apply",
        "https://housing.asu.edu/how-apply-university-housing",
        "housing",
    ),
    _page("housing_current_resident", "https://housing.asu.edu/current-resident", "housing"),
    _page("housing_first_year", "https://housing.asu.edu/first-year-housing", "housing"),
    _page(
        "housing_palo_verde_west",
        "https://housing.asu.edu/housing-communities/residential-colleges/palo-verde-west",
        "housing",
    ),
    # dining
    _page(
        "dining_meal_plan_faq",
        "https://sundevilhospitality.asu.edu/meal-plans/meal-plan-faq",
        "dining",
    ),
    _page(
        "dining_traditional_meal_plans",
        "https://sundevilhospitality.asu.edu/meal-plans/traditional-meal-plans",
        "dining",
    ),
    # health
    _page("health_locations_hours", "https://eoss.asu.edu/health/contact", "health", 72),
    _page("health_home", "https://eoss.asu.edu/health", "health"),
    _page("health_billing", "https://eoss.asu.edu/health/billing-insurance", "health"),
    _page(
        "health_insurance_ship",
        "https://eoss.asu.edu/health/billing-insurance/coverage-options",
        "health",
        72,
    ),
    _page(
        "health_international_insurance",
        "https://eoss.asu.edu/health/billing-insurance/international-students",
        "health",
    ),
    _page("health_immunization", "https://eoss.asu.edu/health/parents/immunization", "health"),
    # counseling
    _page("counseling_home", "https://eoss.asu.edu/counseling", "counseling"),
    _page(
        "counseling_locations_hours",
        "https://eoss.asu.edu/counseling/about-us/location-and-hours",
        "counseling",
    ),
    _page("counseling_crisis", "https://eoss.asu.edu/counseling/services/crisis", "counseling"),
    _page(
        "counseling_open_call_chat",
        "https://eoss.asu.edu/counseling/services/open-call-and-open-chat",
        "counseling",
    ),
    _page(
        "counseling_concerned_student",
        "https://eoss.asu.edu/counseling/concerned",
        "counseling",
    ),
    # recreation
    _page("fitness_join", "https://fitness.asu.edu/join", "recreation"),
    _page("fitness_informal_rec", "https://fitness.asu.edu/facilities/informal-rec", "recreation"),
    _page("fitness_tempe", "https://fitness.asu.edu/facilities/amenities/tempe", "recreation"),
    _page(
        "fitness_intramurals",
        "https://fitness.asu.edu/teams-and-clubs/intramural-sports",
        "recreation",
    ),
    _page("fitness_bike_coop", "https://fitness.asu.edu/amenities/bike-co-op", "recreation"),
    # parking
    _page("pts_permits", "https://cfo.asu.edu/permits", "parking"),
    _page("pts_tempe_permits", "https://cfo.asu.edu/pts-parking-tempe", "parking", 72),
    _page("pts_downtown_permits", "https://cfo.asu.edu/pts-parking-downtown", "parking", 72),
    _page("pts_poly_permits", "https://cfo.asu.edu/pts-parking-poly", "parking"),
    _page("pts_west_permits", "https://cfo.asu.edu/pts-parking-west", "parking"),
    _page("pts_daily_hourly_visitor", "https://cfo.asu.edu/daily-and-hourly", "parking"),
    _page("pts_citations", "https://cfo.asu.edu/parking-citations", "parking"),
    _page("pts_fine_schedule", "https://cfo.asu.edu/violation-code-and-fine-schedule", "parking"),
    _page("pts_hub", "https://cfo.asu.edu/transportation", "parking"),
    # transit
    _page("transit_passes_upass", "https://cfo.asu.edu/transit-passes", "transit"),
    _page("transit_public", "https://cfo.asu.edu/transit", "transit"),
    _page("transit_tempe_west_express", "https://cfo.asu.edu/tempe-west-express", "transit"),
    _page("bike_registration", "https://cfo.asu.edu/bike-registration", "transit"),
    _page("bike_parking", "https://cfo.asu.edu/bike-parking", "transit"),
    # card
    _page("sun_card", "https://cfo.asu.edu/cardservices", "card"),
    _page("mobile_id_faq", "https://cfo.asu.edu/mobile-ID-FAQs", "card"),
    _page("sun_card_office_mu", "https://eoss.asu.edu/mu/whats_in/sun-devil-card-services", "card"),
    # it
    _page("it_experience_center", "https://tech.asu.edu/services/ec", "it"),
    _page(
        "it_connect_network_kb",
        "https://asu.my.salesforce-sites.com/kb/articles/FAQ/How-to-Connect-to-ASU-s-Network",
        "it",
    ),
    _page("it_eduroam", "https://tech.asu.edu/services/campus-it-resources/wifi/eduroam", "it"),
    _page("it_network_faqs", "https://tech.asu.edu/faqs", "it"),
    _page(
        "it_office365",
        "https://asu.my.salesforce-sites.com/kb/articles/FAQ/How-do-I-get-Microsoft-Office-365",
        "it",
    ),
    _page(
        "it_duo_manage_devices",
        "https://asu.my.salesforce-sites.com/kb/articles/FAQ/Manage-Your-Devices-Configured-for-Two-Factor-Authentication/?l=en_US&fs=RelatedArticle",
        "it",
    ),
    _page("it_adobe_cc", "https://tech.asu.edu/adobe-cc", "it"),
    # accessibility
    _page("sails_home", "https://eoss.asu.edu/accessibility", "accessibility"),
    _page(
        "sails_current_student",
        "https://eoss.asu.edu/accessibility/services/current-student",
        "accessibility",
    ),
    _page(
        "sails_future_student",
        "https://eoss.asu.edu/accessibility/services/future-student",
        "accessibility",
    ),
    # safety
    _page("police_home", "https://cfo.asu.edu/police", "safety"),
    _page("police_contact", "https://cfo.asu.edu/police-contact", "safety"),
    _page("police_faq", "https://cfo.asu.edu/police-frequently-asked-questions", "safety"),
    _page("safety_escort", "https://cfo.asu.edu/safety-escort-reservations", "safety"),
    _page("livesafe_app", "https://cfo.asu.edu/livesafe-mobile-app", "safety"),
    _page("emergency_info", "https://cfo.asu.edu/emergency", "safety"),
    # dean_of_students
    _page("title_ix_statement", "https://www.asu.edu/about/title-ix", "dean_of_students"),
    _page("svp_report", "https://sexualviolenceprevention.asu.edu/report", "dean_of_students"),
    _page("svp_faqs", "https://sexualviolenceprevention.asu.edu/faqs", "dean_of_students"),
    _page("dos_home", "https://eoss.asu.edu/dos", "dean_of_students"),
    # basic_needs
    _page(
        "dos_advocacy_crisis_fund",
        "https://eoss.asu.edu/dos/student-advocacy-and-assistance",
        "basic_needs",
    ),
    _page("pitchfork_pantry_get_food", "https://www.pitchforkpantry.org/event", "basic_needs", 72),
    _page("pitchfork_pantry_home", "https://www.pitchforkpantry.org/", "basic_needs"),
    # services
    _page("print_anywhere", "https://print.asu.edu/print-anywhere", "services"),
    _page("childcare_on_campus", "https://eoss.asu.edu/students-families/oncampus", "services"),
    _page("legal_civil_clinic", "https://law.asu.edu/civillegalasst", "services"),
    _page("mail_student_deliveries", "https://cfo.asu.edu/student-deliveries", "services"),
    _page("lost_and_found_mu", "https://eoss.asu.edu/mu/lostandfound", "services"),
    # student_government
    _page("usg_tempe_services", "https://www.asuusgt.org/services", "student_government"),
    _page("asasu_student_gov", "https://eoss.asu.edu/studentgov", "student_government"),
    # campus
    _page("memorial_union", "https://eoss.asu.edu/mu", "campus"),
    _page("student_unions_services", "https://eoss.asu.edu/student-unions/services", "campus"),
    _page("student_unions_overview", "https://eoss.asu.edu/student-unions", "campus"),
    _page("campus_tempe", "https://campus.asu.edu/tempe", "campus"),
    _page("campus_downtown", "https://campus.asu.edu/downtown-phoenix", "campus"),
    _page("campus_polytechnic", "https://campus.asu.edu/polytechnic-campus", "campus"),
    _page("campus_west_valley", "https://campus.asu.edu/west-valley", "campus"),
    _page("campus_lake_havasu", "https://havasu.asu.edu/", "campus"),
    # traditions
    _page("alumni_traditions", "https://alumni.asu.edu/remember/traditions", "traditions"),
    _page("sun_devil_inferno_traditions", "https://eoss.asu.edu/inferno/traditions", "traditions"),
    # about
    _page("asu_facts_figures", "https://www.asu.edu/about/facts-and-figures", "about"),
    _page("asu_innovation_ranking", "https://yourfuture.asu.edu/innovation", "about"),
    _page("asu_rankings", "https://www.asu.edu/about/rankings", "about"),
)
