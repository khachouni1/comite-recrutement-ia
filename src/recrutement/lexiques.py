"""
Lexiques utilisés pour extraire des critères explicables à partir du texte (CV et offres, en anglais).

Chaque compétence est un nom canonique associé à ses variantes d'écriture.
Les lexiques couvrent les métiers présents dans les données : informatique, data, analyse métier,
comptabilité / finance, gestion de projet, santé, ingénierie, électricité, administration.
"""

HARD_SKILLS = {
    # --- informatique & développement
    "python": ["python"], "java": ["java"], "javascript": ["javascript", "js"], "typescript": ["typescript"],
    "c++": ["c\\+\\+"], "c#": ["c#", "\\.net", "dotnet"], "php": ["php"], "ruby": ["ruby"], "go": ["golang"],
    "scala": ["scala"], "kotlin": ["kotlin"], "swift": ["swift"], "r": ["r programming", "rstudio"],
    "html/css": ["html", "css"], "react": ["react", "reactjs"], "angular": ["angular"], "node.js": ["node\\.?js"],
    "django/flask": ["django", "flask"], "spring": ["spring boot", "spring"], "rest api": ["rest", "restful", "api"],
    "microservices": ["microservices?"], "git": ["git", "github", "gitlab"], "linux": ["linux", "unix"],
    "shell scripting": ["bash", "shell scripting", "powershell"], "embedded systems": ["embedded"],
    "testing / qa": ["qa", "quality assurance", "test automation", "selenium", "unit testing", "regression testing"],
    "devops": ["devops", "ci/cd", "jenkins"], "docker": ["docker"], "kubernetes": ["kubernetes", "k8s"],
    "aws": ["aws", "amazon web services"], "azure": ["azure"], "gcp": ["gcp", "google cloud"],
    "cloud": ["cloud"], "networking": ["networking", "tcp/ip", "cisco"], "cybersecurity": ["security", "cybersecurity"],
    "servicenow": ["servicenow"], "salesforce": ["salesforce"], "sap": ["sap"], "oracle": ["oracle"],
    "mainframe": ["mainframe", "cobol"], "agile / scrum": ["agile", "scrum", "kanban", "jira"],
    # --- data
    "sql": ["sql", "mysql", "postgresql", "t-sql", "pl/sql"], "nosql": ["nosql", "mongodb", "cassandra"],
    "excel": ["excel", "spreadsheets?", "vlookup", "pivot tables?"], "power bi": ["power bi"],
    "tableau": ["tableau"], "data analysis": ["data analysis", "analytics", "data analytics"],
    "data visualization": ["data visualization", "dashboards?"], "data modeling": ["data modell?ing"],
    "etl": ["etl", "data pipelines?", "informatica", "ssis"], "data warehouse": ["data warehous\\w*", "snowflake"],
    "big data": ["big data", "hadoop", "spark", "databricks"], "statistics": ["statistics", "statistical"],
    "machine learning": ["machine learning", "deep learning", "ml"], "reporting": ["reporting", "reports"],
    "sas/spss": ["sas", "spss"],
    # --- analyse métier & gestion de projet
    "business analysis": ["business analysis", "business analyst", "requirements gathering", "requirement analysis"],
    "user stories": ["user stories", "use cases"], "uat": ["uat", "user acceptance testing"],
    "process improvement": ["process improvement", "lean", "six sigma"], "project management": ["project management", "pmp", "project manager"],
    "stakeholder management": ["stakeholders?"], "risk management": ["risk management", "risk assessment"],
    "budgeting": ["budget\\w*"], "forecasting": ["forecast\\w*"], "crm": ["crm"], "erp": ["erp"],
    "documentation": ["documentation", "technical writing"], "ms office": ["microsoft office", "ms office", "word", "powerpoint", "outlook"],
    # --- comptabilité & finance
    "accounting": ["accounting", "accountant"], "gaap": ["gaap", "ifrs"], "general ledger": ["general ledger", "gl"],
    "accounts payable": ["accounts payable", "a/p"], "accounts receivable": ["accounts receivable", "a/r"],
    "reconciliation": ["reconcil\\w*"], "month-end close": ["month[- ]end", "closing process", "year[- ]end"],
    "financial statements": ["financial statements?", "financial reporting"], "payroll": ["payroll"],
    "tax": ["tax", "taxation"], "audit": ["audit\\w*"], "quickbooks": ["quickbooks"], "cpa": ["cpa"],
    "financial analysis": ["financial analysis", "financial modell?ing", "variance analysis"], "invoicing": ["invoic\\w*", "billing"],
    # --- santé
    "patient care": ["patient care", "patients?"], "medical terminology": ["medical terminology"],
    "ehr": ["ehr", "emr", "electronic health records?", "epic"], "phlebotomy": ["phlebotomy"], "vital signs": ["vital signs"],
    "hipaa": ["hipaa"], "cpr": ["cpr", "bls"], "laboratory": ["laborator\\w*", "lab"], "clinical": ["clinical"],
    "medical billing": ["medical billing", "medical coding", "icd-10", "cpt"],
    # --- ingénierie & électricité
    "electrical systems": ["electrical"], "nec code": ["national electrical code", "nec"], "plc": ["plc"],
    "autocad": ["autocad", "cad"], "solidworks": ["solidworks"], "blueprints": ["blueprints?", "schematics?"],
    "osha": ["osha"], "maintenance": ["maintenance", "troubleshoot\\w*"], "manufacturing": ["manufacturing"],
    "hvac": ["hvac"], "construction": ["construction"], "quality control": ["quality control", "qc", "inspection"],
    "matlab": ["matlab"], "verification": ["verification", "validation"], "hardware": ["hardware", "fpga", "verilog", "vhdl"],
    # --- commerce, support & administration
    "sales": ["sales", "selling"], "marketing": ["marketing", "seo"], "customer service": ["customer service", "customer support"],
    "data entry": ["data entry"], "scheduling": ["scheduling", "calendar"], "inventory": ["inventory"],
    "logistics": ["logistics", "supply chain", "procurement"], "recruiting": ["recruit\\w*", "talent acquisition"],
    "training": ["training", "onboarding"], "compliance": ["compliance", "regulatory"],
    "security clearance": ["security clearance", "secret clearance", "ts/sci"],
}

SOFT_SKILLS = {
    "communication": ["communication", "communicate", "verbal", "written"],
    "teamwork": ["team ?work", "team player", "collaborat\\w*", "cross[- ]functional"],
    "leadership": ["leadership", "lead", "mentor\\w*", "supervis\\w*"],
    "problem solving": ["problem[- ]solving", "solve problems", "troubleshooting"],
    "analytical thinking": ["analytical", "critical thinking"],
    "attention to detail": ["detail[- ]oriented", "attention to detail", "meticulous"],
    "organization": ["organi[sz]ed", "organizational", "multitask\\w*", "prioritiz\\w*"],
    "time management": ["time management", "deadlines?", "fast[- ]paced"],
    "adaptability": ["adaptab\\w*", "flexib\\w*"], "initiative": ["self[- ]motivated", "proactive", "initiative"],
    "customer focus": ["customer[- ]focused", "client[- ]facing", "customer oriented"],
    "interpersonal": ["interpersonal", "relationship building", "people skills"],
    "creativity": ["creativ\\w*", "innovative"], "negotiation": ["negotiat\\w*"],
    "presentation": ["presentation skills", "presenting", "public speaking"],
    "independence": ["independent\\w*", "autonomous\\w*", "minimal supervision"],
}

# Niveaux de diplôme (ordinaux) : 0 = aucun mentionné … 5 = doctorat
DEGREES = [
    (5, ["ph\\.?d", "doctorate", "doctoral"]),
    (4, ["master'?s?", "m\\.s\\.", "msc", "mba", "m\\.b\\.a"]),
    (3, ["bachelor'?s?", "b\\.s\\.", "b\\.a\\.", "bsc", "undergraduate degree", "4[- ]year degree", "college degree"]),
    (2, ["associate'?s?( degree)?", "2[- ]year degree", "technical diploma", "certificate program"]),
    (1, ["high school", "ged", "diploma"]),
]
DEGREE_NAMES = {0: "non précisé", 1: "bac / lycée", 2: "bac+2", 3: "licence (bachelor)", 4: "master / MBA", 5: "doctorat"}
