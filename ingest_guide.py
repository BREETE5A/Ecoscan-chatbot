"""
ingest_guide.py — Ingestion de toutes les données EcoScan dans ChromaDB
Exécuter une seule fois (ou après mise à jour du guide) :
    python ingest_guide.py
"""

import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

load_dotenv()

CHROMA_PATH   = os.getenv("CHROMA_PATH", "./chroma_ecoscan")
COLLECTION    = os.getenv("COLLECTION_NAME", "ecoscan_knowledge")
EMBED_MODEL   = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# ═══════════════════════════════════════════════════════════════════════════════
# 1. NAVIGATION DE L'APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

NAV_DOCS = [
    # ── Paramètres & profil ───────────────────────────────────────────────────
    ("nav_password", """NAVIGATION APP ECOSCAN — Changer le mot de passe
Pour changer votre mot de passe, allez dans : Profil → Paramètres → Changer le mot de passe.
Route technique : /change-password
How to change password: go to Profile → Settings → Change Password.
لتغيير كلمة المرور: اذهب إلى الملف الشخصي ← الإعدادات ← تغيير كلمة المرور.
Para cambiar la contraseña: ir a Perfil → Ajustes → Cambiar contraseña."""),

    ("nav_email", """NAVIGATION APP ECOSCAN — Changer l'adresse e-mail
Pour changer votre email, allez dans : Profil → Paramètres → Changer l'email.
Route technique : /change-email
How to change email: go to Profile → Settings → Change Email.
لتغيير البريد الإلكتروني: اذهب إلى الملف الشخصي ← الإعدادات ← تغيير البريد الإلكتروني.
Para cambiar el correo: ir a Perfil → Ajustes → Cambiar email."""),

    ("nav_2fa", """NAVIGATION APP ECOSCAN — Authentification à deux facteurs (2FA)
Pour activer ou désactiver la 2FA, allez dans : Profil → Paramètres → Authentification 2 facteurs.
Route technique : /two-factor-setup
To enable 2FA: Profile → Settings → Two-Factor Authentication.
لتفعيل المصادقة الثنائية: الملف الشخصي ← الإعدادات ← المصادقة بخطوتين.
Para activar 2FA: Perfil → Ajustes → Autenticación de dos factores."""),

    ("nav_edit_profile", """NAVIGATION APP ECOSCAN — Modifier le profil
Pour modifier votre nom, photo de profil ou bio, allez dans : Profil → Modifier le profil.
Route technique : /profile/edit
To edit your profile: Profile → Edit Profile.
لتعديل ملفك الشخصي: الملف الشخصي ← تعديل الملف الشخصي.
Para editar el perfil: Perfil → Editar perfil."""),

    ("nav_settings", """NAVIGATION APP ECOSCAN — Paramètres
Les paramètres de l'application (langue, thème, notifications, sécurité) se trouvent dans : Profil → Paramètres.
Route technique : /settings
App settings: Profile → Settings.
إعدادات التطبيق: الملف الشخصي ← الإعدادات.
Ajustes de la app: Perfil → Ajustes."""),

    # ── Fonctionnalités principales ───────────────────────────────────────────
    ("nav_scan", """NAVIGATION APP ECOSCAN — Scanner un déchet (IA photo)
Pour scanner un déchet et obtenir son tri, appuyez sur le bouton central de la barre de navigation (icône caméra).
Route technique : /scan
To scan waste with AI: tap the camera button in the center navigation bar.
لمسح النفايات بالذكاء الاصطناعي: اضغط زر الكاميرا في شريط التنقل الأوسط.
Para escanear un residuo con IA: pulsa el botón de cámara en la barra de navegación central."""),

    ("nav_barcode", """NAVIGATION APP ECOSCAN — Scanner un code-barres produit alimentaire
Pour scanner le code-barres d'un produit (Nutri-Score, Eco-Score), allez dans : Scanner → Code-barres.
Route technique : /barcode
To scan a product barcode: Scan → Barcode.
لمسح الباركود: المسح ← الباركود.
Para escanear código de barras: Escanear → Código de barras."""),

    ("nav_guide", """NAVIGATION APP ECOSCAN — Guide de tri des déchets
Le guide de tri complet est accessible depuis le menu principal → Guide.
Vous y trouverez : 70+ déchets par catégorie, conseils de tri, lois marocaines, contacts des opérateurs.
Route technique : /guide
The sorting guide is in: Menu → Guide.
دليل الفرز: القائمة ← الدليل.
La guía de clasificación: Menú → Guía."""),

    ("nav_map", """NAVIGATION APP ECOSCAN — Carte des points de collecte
Pour trouver les points de collecte ECOVAL, déchetteries et bornes textiles près de chez vous, allez dans : Carte.
Route technique : /map
To find collection points near you: tap the Map tab.
لإيجاد نقاط الجمع القريبة: اضغط على علامة الخريطة.
Para encontrar puntos de recogida: ir a Mapa."""),

    ("nav_stats", """NAVIGATION APP ECOSCAN — Statistiques de recyclage
Pour voir vos statistiques de recyclage (nombre de scans, CO₂ économisé, matériaux), allez dans : Statistiques.
Route technique : /stats
To see your recycling stats: go to the Stats tab.
لرؤية إحصائيات التدوير: اذهب إلى علامة الإحصائيات.
Para ver tus estadísticas de reciclaje: ir a Estadísticas."""),

    ("nav_badges", """NAVIGATION APP ECOSCAN — Mes badges et récompenses
Pour voir vos badges obtenus, allez dans : Profil → Mes badges.
Route technique : /badges
To see your earned badges: Profile → My Badges.
لرؤية الشارات المكتسبة: الملف الشخصي ← شاراتي.
Para ver tus insignias: Perfil → Mis insignias."""),

    ("nav_leaderboard", """NAVIGATION APP ECOSCAN — Classement / Leaderboard
Pour voir le classement des meilleurs recycleurs, allez dans : Classement.
Route technique : /leaderboard
To see the recyclers ranking: go to Leaderboard.
لرؤية ترتيب المدوّرين: اذهب إلى لوحة المتصدرين.
Para ver el ranking de recicladores: ir a Clasificación."""),

    ("nav_challenges", """NAVIGATION APP ECOSCAN — Défis écologiques
Pour voir et relever les défis écologiques, allez dans : Défis.
Route technique : /challenges
To see and complete eco challenges: go to Challenges.
لرؤية التحديات البيئية: اذهب إلى التحديات.
Para ver los retos ecológicos: ir a Retos."""),

    ("nav_rewards", """NAVIGATION APP ECOSCAN — Récompenses et points
Pour voir vos points et récompenses, allez dans : Récompenses.
Route technique : /rewards
To see your points and rewards: go to Rewards.
لرؤية نقاطك ومكافآتك: اذهب إلى المكافآت.
Para ver tus puntos y recompensas: ir a Recompensas."""),

    ("nav_history", """NAVIGATION APP ECOSCAN — Historique des scans
Pour voir l'historique de tous vos scans précédents, allez dans : Profil → Historique.
Route technique : /history
To see your scan history: Profile → History.
لرؤية سجل المسح: الملف الشخصي ← السجل.
Para ver el historial de escaneos: Perfil → Historial."""),

    ("nav_air_quality", """NAVIGATION APP ECOSCAN — Qualité de l'air
Pour voir la qualité de l'air dans votre ville, allez dans : Accueil → Qualité de l'air.
Route technique : /air-quality
To see air quality: Home → Air Quality.
لرؤية جودة الهواء: الرئيسية ← جودة الهواء.
Para ver la calidad del aire: Inicio → Calidad del aire."""),

    ("nav_water", """NAVIGATION APP ECOSCAN — Empreinte eau
Pour calculer et suivre votre empreinte eau, allez dans : Accueil → Empreinte eau.
Route technique : /water-footprint
To track your water footprint: Home → Water Footprint.
لتتبع بصمتك المائية: الرئيسية ← البصمة المائية.
Para tu huella hídrica: Inicio → Huella hídrica."""),

    ("nav_weather", """NAVIGATION APP ECOSCAN — Météo et conseils écologiques
Pour voir la météo avec des conseils écologiques adaptés, allez dans : Accueil → Météo.
Route technique : /weather
For weather and eco tips: Home → Weather.
للطقس ونصائح بيئية: الرئيسية ← الطقس.
Para el tiempo y consejos ecológicos: Inicio → Tiempo."""),

    ("nav_quiz", """NAVIGATION APP ECOSCAN — Quiz écologique
Pour tester vos connaissances sur l'environnement (25 questions), allez dans : Guide → Quiz.
Route technique : /quiz
To take the eco quiz (25 questions): Guide → Quiz.
لأخذ اختبار بيئي (25 سؤال): الدليل ← الاختبار.
Para el quiz ecológico (25 preguntas): Guía → Quiz."""),

    ("nav_marketplace", """NAVIGATION APP ECOSCAN — Marketplace objets recyclés
Pour acheter ou vendre des objets recyclés / seconde main, allez dans : Marketplace.
Route technique : /marketplace
To buy or sell recycled items: go to Marketplace.
لشراء أو بيع المنتجات المعاد تدويرها: اذهب إلى السوق.
Para comprar o vender objetos reciclados: ir a Mercado."""),

    ("nav_my_ads", """NAVIGATION APP ECOSCAN — Mes annonces marketplace
Pour voir et gérer vos annonces dans le marketplace, allez dans : Marketplace → Mes annonces.
Route technique : /marketplace/my
To manage your marketplace listings: Marketplace → My Ads.
لإدارة إعلاناتك: السوق ← إعلاناتي.
Para gestionar tus anuncios: Mercado → Mis anuncios."""),

    ("nav_notifications", """NAVIGATION APP ECOSCAN — Notifications
Pour voir vos notifications et alertes, allez dans : icône cloche en haut de l'écran ou Paramètres → Notifications.
Route technique : /notifications
To see your notifications: bell icon at the top or Settings → Notifications.
لرؤية الإشعارات: أيقونة الجرس في الأعلى أو الإعدادات ← الإشعارات.
Para ver las notificaciones: icono de campana en la parte superior o Ajustes → Notificaciones."""),
]

# ═══════════════════════════════════════════════════════════════════════════════
# 2. CONTACTS OPÉRATEURS DE RECYCLAGE
# ═══════════════════════════════════════════════════════════════════════════════

OPERATOR_DOCS = [
    ("op_averda", """CONTACT OPÉRATEUR RECYCLAGE — AVERDA Maroc (vérifié)
Rôle : Collecte déchets ménagers et nettoiement urbain — National
Téléphone : 0522 92 32 60
Adresse : 213 Bd Anoual, Casablanca
Site : averda.com
Gestion déléguée à Casablanca (16 arrondissements), Rabat et autres villes.
Pour signaler un encombrant à Casablanca, contacter AVERDA ou utiliser l'app Casa Mdinti.
AVERDA Maroc waste collection. Tel: 0522 92 32 60 — Casablanca. averda.com
جمع النفايات الحضرية. هاتف: 0522 92 32 60 — الدار البيضاء."""),

    ("op_arma", """CONTACT OPÉRATEUR RECYCLAGE — ARMA Environnement (vérifié)
Rôle : Collecte et propreté urbaine — Rabat / Casablanca
Téléphone : 0537 66 32 90
Adresse : Av. Mehdi Ben Barka, Lot Soumia N°5, Souissi, Rabat
Email : contact@arma.ma — arma.ma
Opérateur officiel de collecte à Rabat depuis 2015.
Pour signaler un encombrant à Rabat, contacter ARMA directement.
ARMA Environnement Rabat. Tel: 0537 66 32 90. contact@arma.ma — arma.ma
ARMA بيئة الرباط. هاتف: 0537 66 32 90. contact@arma.ma"""),

    ("op_suez", """CONTACT OPÉRATEUR RECYCLAGE — Suez El Beida (anciennement SITA El Beida)
Rôle : Collecte ordures ménagères — Multi-villes
Filiale groupe Suez Environnement.
Opère à Casablanca, Rabat, Mohammedia, El Jadida, Oujda, Tétouan.
Collecte et traitement des déchets ménagers. Filiale "SITA Maroc Recycling" pour le recyclage.
Suez El Beida (formerly SITA El Beida) household waste. Casablanca, Rabat, Mohammedia.
سويز البيضاء (سابقاً SITA البيضاء) لجمع النفايات المنزلية. الدار البيضاء والرباط."""),

    ("op_tecmed", """CONTACT OPÉRATEUR RECYCLAGE — TECMED Maroc (vérifié)
Rôle : Collecte urbaine — Tanger, Rabat
Téléphone : 0539 95 85 89
Adresse : Z.I. Al Majd, Tanger
Filiale groupe Urbaser (Espagne).
Collecte des déchets ménagers et nettoiement urbain à Tanger et autres villes.
TECMED Maroc urban collection Tangier. Tel: 0539 95 85 89.
تكميد المغرب للجمع الحضري طنجة. هاتف: 0539 95 85 89."""),

    ("op_smrr", """CONTACT OPÉRATEUR RECYCLAGE — SMRR (vérifié)
Rôle : Recyclage papier, carton, PET — National
Téléphone : 0522 51 09 94
Adresse : Zone Ind. Tit Mellil, Casablanca
Email : adm@smrr.ma — smrr.ma
Société Marocaine de Récupération et de Recyclage. Leader du recyclage papier/carton/PET.
Présent à Casablanca, Tanger, Agadir, Marrakech.
SMRR Moroccan recycling leader — paper, cardboard, PET. Tel: 0522 51 09 94. smrr.ma
SMRR رائدة التدوير المغربية — ورق، كرتون، PET. هاتف: 0522 51 09 94."""),

    ("op_ecozalagh", """CONTACT OPÉRATEUR RECYCLAGE — Eco Zalagh (vérifié)
Rôle : DEEE (appareils électroniques) — National
Téléphone : 0661 51 47 11
Email : Contact@ecozalagh.com — ecozalagh.com
Adresse : Sidi Hajaj Oulad Mrah, Casablanca
Spécialiste collecte et recyclage DEEE pour particuliers et entreprises.
Couvre : Casablanca, Rabat, Tanger, Fès, Meknès, Marrakech.
Pour déposer un téléphone, ordinateur, TV usagé, contacter Eco Zalagh.
Eco Zalagh WEEE specialist. Tel: 0661 51 47 11. Contact@ecozalagh.com — ecozalagh.com
Eco Zalagh متخصص DEEE. هاتف: 0661 51 47 11."""),

    ("op_ecoval_holcim", """CONTACT OPÉRATEUR — ECOVAL (Holcim Maroc) (vérifié)
Rôle : Valorisation déchets industriels — Région Casablanca
Téléphone : 0522 58 20 05
Adresse : El Gara, commune Ouled Zidane (40 km de Casablanca)
ATTENTION : ECOVAL Holcim est spécialisé dans les déchets INDUSTRIELS.
Ce n'est PAS un service de reprise pour les particuliers (smartphones, etc.).
Les bacs ECOVAL en magasin (Marjane, Carrefour) sont gérés par le programme de collecte sélective.
ECOVAL (Holcim): industrial waste only, NOT household take-back. Tel: 0522 58 20 05.
ECOVAL (هولسيم): للنفايات الصناعية فقط، وليس للأسر. هاتف: 0522 58 20 05."""),

    ("op_zerozabel", """CONTACT ONG — Zéro Zbel (vérifié)
Rôle : Sensibilisation et tri sélectif — National
Email : contact@zerozabel.ma
ONG pionnière du tri sélectif au Maroc.
Sensibilisation dans écoles, entreprises et communes à travers le Maroc.
Programmes de compostage communautaire.
Zero Zbel NGO sorting awareness Morocco. Email: contact@zerozabel.ma.
جمعية زيرو زبل للتوعية بالفرز. البريد: contact@zerozabel.ma"""),

    ("op_encombrants", """CONTACT COLLECTE ENCOMBRANTS (matelas, meubles volumineux)
Casablanca : Contacter AVERDA au 0522 92 32 60 ou utiliser l'application "Casa Mdinti" (Android/iOS).
Rabat : Contacter ARMA Environnement au 0537 66 32 90 ou email contact@arma.ma
Note : Il n'existe pas de numéro unique national pour les encombrants.
Le service varie selon l'opérateur de votre arrondissement/commune.
Casablanca bulky waste: AVERDA 0522 92 32 60 or app "Casa Mdinti".
Rabat bulky waste: ARMA 0537 66 32 90 — contact@arma.ma
الدار البيضاء: AVERDA 0522 92 32 60 أو تطبيق Casa Mdinti.
الرباط: ARMA 0537 66 32 90 — contact@arma.ma"""),
]

# ═══════════════════════════════════════════════════════════════════════════════
# 3. GUIDE DE TRI — ITEMS COMPLETS
# ═══════════════════════════════════════════════════════════════════════════════

GUIDE_ITEMS = [
    # ── MÉTAL ────────────────────────────────────────────────────────────────
    ("item_alu_can", "Métal", "OUI", "Bac jaune ECOVAL",
     "Canette aluminium (Coca-Cola, Pepsi, Fanta) | Aluminum can | علبة ألمنيوم",
     "Écraser pour réduire le volume. Rincer. L'alu est recyclé à l'infini — 8kg CO₂ économisés/kg. "
     "Crush to reduce volume. Rinse. Aluminium saves 8 kg CO₂/kg. "
     "اسحق لتقليل الحجم. اشطف. 8 كجم CO₂ موفرة لكل كجم."),

    ("item_steel_can", "Métal", "OUI", "Bac jaune ECOVAL",
     "Boîte conserve acier (Aïcha, Diamanor) | Steel food can",
     "Rincer. Le couvercle peut rester. Accepté dans tous les bacs jaunes ECOVAL au Maroc. "
     "Rinse. The lid can stay. Accepted in all ECOVAL yellow bins in Morocco."),

    ("item_metal_cap", "Métal", "OUI", "Bac jaune ECOVAL",
     "Capsule métallique (bouteille, bocal) | Metal cap",
     "Petites capsules : glisser dans une boîte conserve vide. "
     "Small caps: slip into an empty can to prevent them falling in sorting machines."),

    ("item_alu_tray", "Métal", "OUI", "Bac jaune ECOVAL",
     "Barquette aluminium (cuisine, tajine) | Aluminum tray",
     "Rincer si possible. Même légèrement souillée, reste recyclable. Écraser en boule. "
     "Rinse if possible. Even slightly soiled, still recyclable. Crush into a ball."),

    ("item_alu_foil", "Métal", "OUI", "Bac jaune ECOVAL",
     "Papier aluminium froissé en boule | Crumpled aluminum foil | ورق ألمنيوم مجعّد",
     "Froisser en boule compacte (min 5cm). Si très souillé de nourriture → poubelle. "
     "Crumple into a compact ball (min 5 cm). If heavily soiled → trash."),

    ("item_metal_lid", "Métal", "OUI", "Bac jaune ECOVAL",
     "Couvercle métallique (pot, bocal) | Metal lid",
     "Séparer du bocal en verre → bac jaune. Le bocal verre → colonne verte. "
     "Separate from glass jar → yellow bin. Glass jar → green column."),

    ("item_aerosol_empty", "Métal", "OUI", "Bac jaune ECOVAL",
     "Aérosol vide dégazé (déodorant, laque) | Empty deflated aerosol",
     "OBLIGATOIREMENT vide et dégazé. Ne jamais percer. "
     "MUST be empty and deflated. Never pierce."),

    ("item_metal_tin", "Métal", "OUI", "Bac jaune ECOVAL",
     "Boîte biscuits métal (Bimo, LU) | Metal biscuit tin",
     "Retirer les emballages papier intérieurs. Écraser si possible. "
     "Remove inner paper wrappings. Crush if possible."),

    ("item_scrap_metal", "Métal", "OUI", "Déchetterie — zone métal",
     "Ferraille / Casserole acier usagée | Scrap metal",
     "Déchetterie zone métaux ou ferracha. Refondu dans la sidérurgie marocaine. "
     "Recycling center metal zone or ferracha."),

    ("item_battery_alkaline", "Métal", "NON", "Collecteur piles",
     "Pile alcaline (AA, AAA, 9V) | Alkaline battery | بطارية قلوية",
     "JAMAIS au bac jaune ! Contient mercure, cadmium. "
     "Déposer dans les collecteurs en supermarché (Marjane, Carrefour, Label'Vie). "
     "NEVER in the yellow bin! Contains mercury, cadmium. Drop at supermarket collectors."),

    ("item_aerosol_full", "Métal", "NON", "Déchetterie — dangereux",
     "Aérosol non vide / sous pression | Non-empty aerosol",
     "Risque d'explosion. Déchetterie zone produits dangereux. Ne jamais percer. "
     "Risk of explosion. Never pierce or throw in regular trash."),

    ("item_paint_can", "Métal", "NON", "Déchetterie — dangereux",
     "Boîte peinture métal avec résidus | Metal paint can",
     "Les résidus contaminent. Déchetterie. Si complètement sèche et vide → bac jaune. "
     "Paint residues contaminate. If completely dry and empty → yellow bin."),

    # ── VERRE ─────────────────────────────────────────────────────────────────
    ("item_glass_bottle", "Verre", "OUI", "Colonne verte",
     "Bouteille verre (Sidi Ali, Ain Atlas, Oulmes) | Glass bottle | زجاجة زجاجية",
     "Rincer. Ne pas casser. Bouchon métallique peut rester. Colonne verte = uniquement verre creux. "
     "Rinse. Do not break. Green column = hollow glass only."),

    ("item_glass_jar", "Verre", "OUI", "Colonne verte",
     "Bocal confiture / conserve (Aïcha) | Jam jar | برطمان",
     "Rincer rapidement. Couvercle métal → bac jaune. Bocal verre → colonne verte. "
     "Quick rinse. Metal lid → yellow bin. Glass jar → green column."),

    ("item_pyrex", "Verre", "NON", "Déchetterie",
     "Pyrex / plat en verre réfractaire | Pyrex | باير",
     "Borosilicate : point de fusion différent. Contamine le lot recyclé. Déchetterie ou don. "
     "Different melting point. Contaminates recycled glass. Recycling center."),

    ("item_mirror", "Verre", "NON", "Déchetterie",
     "Miroir | Mirror | مرآة",
     "Composition chimique différente (revêtement argenté). Non recyclable avec le verre. "
     "Different chemical composition (silver coating). Not recyclable with glass."),

    ("item_broken_glass", "Verre", "NON", "Déchetterie (carton épais)",
     "Verre brisé / cassé | Broken glass | زجاج مكسور",
     "DANGER : blesse les agents de tri. Envelopper dans du carton épais. Jamais dans la colonne verte. "
     "DANGER: injures sorting workers. Wrap in thick cardboard. Never in the green column."),

    ("item_bulb", "Verre", "NON", "Point luminaire",
     "Ampoule (toute technologie) | Light bulb | مصباح",
     "Points de collecte : Bricoma, Mr. Bricolage, déchetterie. LED et fluocompactes = DEEE. "
     "Collection points: Bricoma, Mr. Bricolage. LED and CFL = WEEE."),

    # ── ORGANIQUE ─────────────────────────────────────────────────────────────
    ("item_peels", "Organique", "OUI", "Composteur / Bac marron",
     "Épluchures fruits et légumes | Fruit peels | قشور الفواكه",
     "65% des déchets marocains. Idéal pour compost maison. Se décompose en 2-6 mois. "
     "65% of Moroccan waste. Ideal for home compost. Decomposes in 2-6 months."),

    ("item_coffee", "Organique", "OUI", "Composteur",
     "Marc de café | Coffee grounds | تفل القهوة",
     "Excellent activateur de compost. Mélanger avec des matières sèches (feuilles mortes). "
     "Excellent compost activator. Mix with dry materials."),

    ("item_bread", "Organique", "OUI", "Composteur",
     "Restes de pain / harcha / msemen | Leftover bread",
     "Déchirer en petits morceaux. Peut attirer des oiseaux dans un composteur ouvert. "
     "Tear into small pieces. May attract birds in an open composter."),

    ("item_eggshell", "Organique", "OUI", "Composteur",
     "Coquilles d'œuf | Eggshells | قشور البيض",
     "Écraser avant de composter — apportent du calcium. Se décomposent lentement (1 an). "
     "Crush before composting — provide calcium. Decompose slowly (1 year)."),

    ("item_meat", "Organique", "NON", "Poubelle ordinaire",
     "Viande et poisson (restes) | Meat and fish | بقايا اللحم",
     "Attire rongeurs et insectes dans un composteur ouvert. Poubelle ordinaire. "
     "Attracts rodents. Regular trash."),

    ("item_cooking_oil", "Organique", "NON", "Collecte spéciale huile",
     "Huile de cuisson usagée | Used cooking oil | زيت الطهي المستعمل",
     "JAMAIS dans l'évier (bouche les canalisations). Solidifier dans un contenant fermé → poubelle. "
     "NEVER down the sink. Solidify in a sealed container → trash."),

    ("item_diapers", "Organique", "NON", "Poubelle ordinaire",
     "Couches bébé / protection hygiénique | Baby diapers | حفاضات",
     "Non recyclables et non compostables. Poubelle ordinaire. "
     "Non-recyclable and non-compostable. Regular trash."),

    # ── ÉLECTRONIQUE ──────────────────────────────────────────────────────────
    ("item_smartphone", "Électronique", "COLLECTE SPÉCIALE", "Éco-point ECOVAL",
     "Smartphone / Téléphone portable | Mobile phone | هاتف ذكي",
     "Décret 2-14-782 : reprise obligatoire. ECOVAL rembourse jusqu'à 500 DH. "
     "Points ECOVAL : Marjane, Carrefour, Cosmos Electro. "
     "Mandatory take-back. ECOVAL reimburses up to 500 DH."),

    ("item_computer", "Électronique", "COLLECTE SPÉCIALE", "Éco-point ECOVAL / Déchetterie",
     "Ordinateur / Laptop | Computer | حاسوب",
     "Effacer le disque dur avant dépôt. ECOVAL ou déchetterie zone DEEE. "
     "Erase hard drive before drop-off. ECOVAL or recycling center WEEE zone."),

    ("item_tv", "Électronique", "COLLECTE SPÉCIALE", "Éco-point ECOVAL",
     "Télévision / Écran PC | TV / Screen | تلفزيون",
     "Reprise 1 pour 1 chez le vendeur à l'achat d'un neuf. "
     "1-for-1 take-back when buying new (legal obligation)."),

    ("item_fridge", "Électronique", "COLLECTE SPÉCIALE", "Éco-point ECOVAL",
     "Réfrigérateur / Congélateur | Fridge | ثلاجة",
     "ECOVAL organise la collecte à domicile. Contient des frigorigènes à traiter. "
     "ECOVAL arranges home collection. Contains refrigerants."),

    ("item_liion", "Électronique", "COLLECTE SPÉCIALE", "Collecteur ECOVAL",
     "Pile / Batterie Li-ion (téléphone, vélo) | Li-ion battery | بطارية ليثيوم",
     "JAMAIS à la poubelle (risque d'incendie). Collecteurs ECOVAL, Marjane, Label'Vie. "
     "NEVER in the trash (fire risk). ECOVAL, Marjane, Label'Vie collectors."),

    ("item_charger", "Électronique", "COLLECTE SPÉCIALE", "Éco-point ECOVAL",
     "Chargeur / Câble USB / Adaptateur | Charger | شاحن",
     "Regrouper dans un sac. Éco-point ECOVAL ou magasin électronique (Biougnach, Electroplanet). "
     "Bundle in a bag. ECOVAL point or electronics store."),

    # ── TEXTILE ───────────────────────────────────────────────────────────────
    ("item_clothes_good", "Textile", "OUI", "Borne textile / Souk fripe",
     "Vêtements en bon état | Clothing in good condition | ملابس بحالة جيدة",
     "Déposer propres dans un sac en borne textile. Tradition sadaqa : don à associations "
     "(Caritas Maroc, Croix-Rouge). "
     "Drop off clean. Sadaqa tradition: donate to charities."),

    ("item_shoes", "Textile", "OUI", "Borne textile",
     "Chaussures (toutes matières) | Shoes | أحذية",
     "Attacher par paires avec un lacet. Même très usées — valorisées ou recyclées. "
     "Tie in pairs. Even very worn — valued or recycled."),

    ("item_mattress", "Textile", "NON", "Déchetterie / Collecte encombrants",
     "Matelas / oreillers | Mattress | مرتبة",
     "Trop volumineux pour les bornes. Collecte encombrants commune "
     "(Casablanca 0801003737, Rabat 0537726364) ou déchetterie. "
     "Too bulky. Municipality bulky waste collection: Casablanca 0801003737, Rabat 0537726364."),

    ("item_burnous", "Textile", "OUI", "Borne textile / Association",
     "Burnous / Djellaba usagée | Used burnous djellaba | برنوس / جلابة",
     "Tradition marocaine de sadaqa. Associations ou bornes textile. La laine est recyclable. "
     "Moroccan sadaqa tradition. Wool is recyclable."),

    # ── PLASTIQUE ─────────────────────────────────────────────────────────────
    ("item_pet_bottle", "Plastique", "OUI", "Bac jaune ECOVAL",
     "Bouteille PET (Sidi Ali, Ain Atlas, Pepsi) | PET bottle | زجاجة PET",
     "Rincer. Écraser. Bouchon peut rester vissé. L'un des plastiques les plus recyclés au Maroc. "
     "Rinse. Crush. Cap can stay. One of the most recycled plastics in Morocco."),

    ("item_hdpe_bottle", "Plastique", "OUI", "Bac jaune ECOVAL",
     "Flacon PEHD (shampoing, lait) | HDPE bottle | زجاجة PEHD",
     "Vider et rincer. Écraser. Le PEHD (code 2) est très recyclable. "
     "Empty and rinse. Crush. HDPE (code 2) is very recyclable."),

    ("item_yogurt_pot", "Plastique", "OUI", "Bac jaune ECOVAL",
     "Pot de yaourt (Danone Jaouda, Yawmi) | Yogurt pot | كوب زبادي",
     "Rincer rapidement. Le couvercle alu → bac jaune aussi. "
     "Quick rinse. Aluminum lid → yellow bin too."),

    ("item_plastic_bag", "Plastique", "INTERDIT", "Interdit — Loi 77-15",
     "Sac plastique (tous types) | Plastic bag | كيس بلاستيكي",
     "INTERDIT en production et distribution depuis juillet 2016 (Loi 77-15). "
     "Amende : 1000 à 1 million DH. Utiliser sac tissu ou panier. "
     "BANNED since July 2016 (Law 77-15). Fine: 1,000 to 1 million DH."),

    ("item_soft_film", "Plastique", "NON", "Borne dédiée / Poubelle",
     "Film plastique souple | Soft plastic film | غشاء بلاستيكي مرن",
     "Non recyclable dans les bacs jaunes — bloque les machines de tri. "
     "Certains Carrefour ont des bornes dédiées. "
     "Not recyclable in yellow bins — clogs sorting machines."),

    ("item_polystyrene", "Plastique", "NON", "Déchetterie",
     "Polystyrène expansé | Expanded polystyrene | البوليسترين",
     "Difficile à recycler au Maroc. Volume énorme pour peu de matière. "
     "Difficult to recycle in Morocco. Major pollution issue."),

    ("item_toothpaste", "Plastique", "NON", "Poubelle ordinaire",
     "Tube dentifrice (Colgate, Signal) | Toothpaste tube | معجون أسنان",
     "Multicouche alu + plastique = non recyclable. Programme de reprise Colgate en pharmacies. "
     "Multilayer = not recyclable. Colgate take-back in partner pharmacies."),

    # ── PAPIER / CARTON ───────────────────────────────────────────────────────
    ("item_newspaper", "Papier", "OUI", "Bac bleu / Ferracha",
     "Journal / Magazine | Newspaper | جريدة",
     "Plier ou empiler. Ne pas mouiller. La ferracha joue un rôle essentiel au Maroc. "
     "Fold or stack. Do not wet. The ferracha plays a key role in Morocco."),

    ("item_cardboard", "Papier", "OUI", "Bac bleu / Ferracha",
     "Carton ondulé aplati | Flattened cardboard | كرتون",
     "Aplatir. Enlever agrafes et ruban. Repris par l'industrie papetière (CMCP). "
     "Flatten. Remove staples and tape. Collected by Moroccan paper industry."),

    ("item_food_carton", "Papier", "OUI", "Bac bleu",
     "Brique alimentaire (Centrale Laitière, Candia) | Food carton | عبوة حليب",
     "Rincer. Écraser. Bac bleu ou bac jaune selon la ville — vérifier auprès de la commune. "
     "Rinse. Crush. Blue or yellow bin depending on city."),

    ("item_receipt", "Papier", "NON", "Poubelle ordinaire",
     "Ticket de caisse thermique | Thermal receipt | إيصال",
     "Contient du Bisphénol A (BPA) — perturbateur endocrinien. JAMAIS au bac bleu. "
     "Contains Bisphenol A. NEVER in the blue bin."),

    ("item_pizza_box", "Papier", "NON", "Composteur (si non imprimé)",
     "Carton pizza gras / souillé | Greasy pizza box | علبة بيتزا دهنية",
     "Partie grasse contamine. Découper : partie propre → bac bleu, grasse → compost. "
     "Cut: clean part → blue bin, greasy part → compost or waste."),

    ("item_paper_tissue", "Papier", "NON", "Poubelle ordinaire",
     "Papier hygiénique / mouchoir usagé | Used tissue | ورق صحي",
     "Souillé de matières biologiques → non recyclable. Poubelle ordinaire. "
     "Contaminated with biological material → not recyclable."),

    # ── DÉCHETS DANGEREUX ─────────────────────────────────────────────────────
    ("item_medication", "Dangereux", "COLLECTE SPÉCIALE", "Pharmacie agréée",
     "Médicaments périmés | Expired medications | أدوية منتهية الصلاحية",
     "JAMAIS à la poubelle ni dans les toilettes (contamination eau). Rapporter en pharmacie. "
     "NEVER in the bin or toilet. Return to pharmacy."),

    ("item_motor_oil", "Dangereux", "COLLECTE SPÉCIALE", "Garage agréé",
     "Huile moteur usagée | Used motor oil | زيت محرك مستعمل",
     "Ne jamais verser dans les égouts (1L pollue 1 million L d'eau). Garage agréé. "
     "Never pour down drains. 1L pollutes 1 million L of water."),

    ("item_paint_dangerous", "Dangereux", "COLLECTE SPÉCIALE", "Déchetterie — dangereux",
     "Peinture / Vernis / Solvant | Paint Varnish Solvent | طلاء",
     "Garder dans le contenant d'origine avec l'étiquette. Loi 28-00 : infraction si abandon. "
     "Keep in original container. Law 28-00: criminal offense if abandoned."),

    ("item_syringe", "Dangereux", "COLLECTE SPÉCIALE", "Boîte DASRI / Pharmacie",
     "Seringues / Aiguilles (DASRI) | Syringes | حقن",
     "Déchets d'Activités de Soins à Risques Infectieux. Boîtes DASRI en pharmacie. "
     "Infectious Healthcare Waste. DASRI boxes at pharmacy."),

    ("item_asbestos", "Dangereux", "COLLECTE SPÉCIALE", "Déchetterie — amiante",
     "Amiante (plaques, tuiles) | Asbestos | أسبستوس",
     "TRÈS DANGEREUX — fibre cancérigène. Ne jamais casser ni poncer. "
     "Contacter la commune ou l'ONEE. VERY DANGEROUS. Contact municipality or ONEE."),

    ("item_neon", "Dangereux", "COLLECTE SPÉCIALE", "Point luminaire / Déchetterie",
     "Néons / Tubes fluorescents | Neon tubes | أنابيب نيون",
     "Contiennent du mercure. Ne jamais casser. Bricoma, Mr. Bricolage ou déchetterie. "
     "Contain mercury. Never break. Collection at Bricoma, Mr. Bricolage."),

    # ── VERRE — items manquants ───────────────────────────────────────────────
    ("item_baby_jar", "Verre", "OUI", "Colonne verte",
     "Pot bébé (Blédina, Nestlé Bébé) | Baby food jar | برطمان غذاء الأطفال",
     "Rincer. Très recyclable. Même les petits pots sont acceptés. "
     "Colonne verte dans les villes (Casablanca, Rabat, Marrakech). "
     "Very recyclable. Even small jars accepted in green columns."),

    ("item_perfume_bottle", "Verre", "OUI", "Colonne verte",
     "Flacon parfum en verre | Glass perfume bottle | قارورة عطر زجاجية",
     "Vider complètement. Retirer la pompe plastique si possible (va à la poubelle ordinaire). "
     "Le verre → colonne verte. Empty completely. Remove plastic pump → regular trash."),

    ("item_glass_carafe", "Verre", "OUI", "Colonne verte",
     "Carafe / vase en verre ordinaire | Glass carafe vase | إبريق / مزهرية زجاجية",
     "Verre ordinaire recyclable. Ne pas confondre avec cristal (non recyclable). "
     "Déposer doucement pour éviter la casse. Ordinary glass is recyclable. Not crystal."),

    ("item_honey_jar", "Verre", "OUI", "Colonne verte",
     "Pot de miel en verre | Glass honey jar | برطمان عسل زجاجي",
     "Rincer à l'eau tiède pour décoller le miel. Couvercle métallique → bac jaune. "
     "Accepté dans toutes les colonnes vertes. Rinse with warm water. Metal lid → yellow bin."),

    ("item_tempered_glass", "Verre", "NON", "Déchetterie",
     "Verre trempé (pare-brise, vitre auto) | Tempered glass windshield | زجاج مقسّى",
     "Traitement thermique spécial incompatible avec le recyclage du verre ordinaire. "
     "Casse en miettes → contamination. Déchetterie. "
     "Special heat treatment incompatible with ordinary glass recycling. Recycling center."),

    ("item_ceramic", "Verre", "NON", "Déchetterie",
     "Faïence / céramique / porcelaine | Earthenware ceramic porcelain | فخار / سيراميك",
     "Pas du verre ! Composition totalement différente. Déchetterie ou ressourcerie si intact. "
     "Jamais dans la colonne verte. Not glass! Never in the green column."),

    # ── ORGANIQUE — items manquants ───────────────────────────────────────────
    ("item_tea_bags", "Organique", "OUI", "Composteur",
     "Sachets thé usagés (Lipton, Atay Baraka) | Used tea bags | أكياس شاي مستعملة",
     "Enlever les agrafes métalliques avant de composter. "
     "Certains sachets synthétiques (nylon) → poubelle ordinaire. "
     "Remove metal staples before composting. Synthetic nylon bags → regular trash."),

    ("item_leaves", "Organique", "OUI", "Composteur",
     "Feuilles mortes / gazon / herbes | Dead leaves lawn clippings | أوراق جافة / عشب",
     "Mélanger vert (azote : feuilles fraîches, épluchures) et sec (carbone : feuilles mortes, carton). "
     "Rapport idéal 1/3 vert, 2/3 sec. Mix green (nitrogen) and dry (carbon). Ideal ratio: 1/3 green, 2/3 dry."),

    ("item_sawdust", "Organique", "OUI", "Composteur",
     "Sciure de bois non traité | Untreated sawdust | نشارة خشب غير معالج",
     "Bon apport de carbone pour le compost. Uniquement bois naturel non peint, non verni. "
     "Pas de panneaux agglomérés (colle chimique). "
     "Good carbon input. Only untreated, unpainted wood. No chipboard."),

    ("item_cardboard_compost", "Organique", "OUI", "Composteur",
     "Carton déchiré non imprimé (pour compost) | Torn unprinted cardboard | كرتون ممزق غير مطبوع",
     "En petits morceaux humidifiés. Apport de carbone. "
     "Éviter carton imprimé (encres) ou ciré. Préférer le bac bleu si propre. "
     "In small dampened pieces. Avoid printed or waxed cardboard."),

    ("item_dairy", "Organique", "NON", "Poubelle ordinaire",
     "Produits laitiers (yaourt, fromage) | Dairy products yogurt cheese | منتجات الألبان",
     "Fermentation rapide = odeurs et nuisibles. Poubelle ordinaire. "
     "Les emballages plastiques/carton → bacs recyclage. "
     "Fast fermentation = odors and pests. Regular trash. Plastic packaging → recycling bins."),

    ("item_ashes", "Organique", "NON", "Poubelle ordinaire",
     "Cendres froides (barbecue, cheminée) | Cold ashes barbecue fireplace | رماد بارد",
     "En très petite quantité dans le compost si froid (apport de potasse). "
     "En grande quantité → poubelle. Jamais chaude. "
     "In very small cold amounts in compost (potassium). In large quantities → trash. Never hot."),

    # ── ÉLECTRONIQUE — items manquants ────────────────────────────────────────
    ("item_tablet", "Électronique", "COLLECTE SPÉCIALE", "Éco-point ECOVAL",
     "Tablette / iPad | Tablet iPad | لوحي / آيباد",
     "Effacer les données avant dépôt (réinitialisation usine). "
     "Contient lithium, coltan, or — métaux précieux valorisables. "
     "Erase data before drop-off (factory reset). Contains lithium, coltan, gold."),

    ("item_washing_machine", "Électronique", "COLLECTE SPÉCIALE", "Éco-point ECOVAL",
     "Machine à laver / Lave-vaisselle | Washing machine dishwasher | غسالة",
     "Reprise possible par le vendeur. ECOVAL organise collecte à domicile. "
     "Récupération du cuivre des moteurs et de l'acier inoxydable. "
     "Take-back by retailer. ECOVAL arranges collection. Copper and stainless steel recovered."),

    ("item_aircon", "Électronique", "COLLECTE SPÉCIALE", "Technicien agréé",
     "Climatiseur / Split (unité intérieure) | Air conditioner split | مكيف هواء",
     "Le frigorigène doit être récupéré par un technicien agréé avant démontage. "
     "Puis éco-point ECOVAL pour le reste. "
     "Refrigerant must be recovered by a certified technician before dismantling."),

    ("item_led_bulb", "Électronique", "COLLECTE SPÉCIALE", "Point luminaire",
     "Ampoule LED / Fluocompacte | LED CFL light bulb | مصباح LED",
     "Points de collecte : Bricoma, Mr. Bricolage, grandes surfaces. "
     "Contient mercure (fluocompactes) → manipulation délicate. "
     "Collection points: Bricoma, Mr. Bricolage. CFL bulbs contain mercury."),

    ("item_printer", "Électronique", "COLLECTE SPÉCIALE", "Éco-point ECOVAL",
     "Imprimante / Cartouche d'encre | Printer ink cartridge | طابعة / خرطوشة حبر",
     "Cartouches : programmes de reprise fabricant (HP, Epson, Canon). "
     "Imprimantes complètes → éco-point ECOVAL. "
     "Cartridges: manufacturer take-back (HP, Epson, Canon). Full printers → ECOVAL."),

    ("item_microwave", "Électronique", "COLLECTE SPÉCIALE", "Éco-point ECOVAL",
     "Four micro-ondes | Microwave oven | فرن الميكروويف",
     "Contient des composants électroniques et magnétrons. "
     "Éco-point ECOVAL ou reprise vendeur. Ne jamais démonter soi-même. "
     "Contains electronic components and magnetrons. Never disassemble yourself."),

    ("item_vacuum", "Électronique", "COLLECTE SPÉCIALE", "Éco-point ECOVAL",
     "Aspirateur / Petit électroménager | Vacuum cleaner small appliance | مكنسة كهربائية",
     "Tous les appareils électriques sont concernés par le Décret 2-14-782. "
     "Éco-point ECOVAL ou déchetterie zone DEEE. "
     "All electrical appliances covered by Decree 2-12-695."),

    # ── TEXTILE — items manquants ─────────────────────────────────────────────
    ("item_clothes_damaged", "Textile", "OUI", "Borne textile",
     "Vêtements abîmés / déchirés | Damaged torn clothing | ملابس تالفة / ممزقة",
     "Même abîmés, recyclés en chiffons industriels, isolants thermiques ou couvertures. "
     "Ne pas jeter à la poubelle. Even damaged, recycled into industrial rags or insulation."),

    ("item_fabric_bag", "Textile", "OUI", "Borne textile / Association",
     "Sac en tissu / sac à main tissu | Fabric bag handbag | حقيبة قماشية",
     "Nettoyer si possible. Borne textile ou association. "
     "Alternative au sac plastique à encourager (Loi 77-15 depuis 2016). "
     "Clean if possible. Alternative to plastic bags (Law 77-15 since 2016)."),

    ("item_bed_linen", "Textile", "OUI", "Borne textile",
     "Linge de maison (draps, serviettes, rideaux) | Household linen sheets towels | مفروشات ملاءات",
     "Propre et sec. Borne textile ou ressourcerie. "
     "Les draps usagés deviennent des chiffons industriels pour les garages. "
     "Clean and dry. Used sheets become industrial rags for garages."),

    ("item_stuffed_toys", "Textile", "OUI", "Borne textile / Association",
     "Peluches / jouets tissu propres | Clean stuffed toys | دمى / ألعاب قماشية نظيفة",
     "Propres et sans piles. Associations caritatives (Bayti, SOS Villages d'Enfants Maroc) "
     "ou bornes textile. Clean and battery-free. Charities: Bayti, SOS Villages."),

    ("item_leather", "Textile", "OUI", "Borne textile / Souk fripe",
     "Ceinture cuir / sac cuir | Leather belt bag | حزام جلدي / حقيبة جلدية",
     "Souk fripe marocain (Derb Ghallef, Mellah de Marrakech) ou bornes textile. "
     "Le cuir a une longue durée de vie. "
     "Moroccan flea market (Derb Ghallef, Marrakech Mellah) or textile bins."),

    ("item_underwear", "Textile", "NON", "Poubelle ordinaire",
     "Sous-vêtements usagés | Used underwear | ملابس داخلية مستعملة",
     "Pour des raisons d'hygiène, non acceptés dans les bornes textile ni les associations. "
     "Poubelle ordinaire. For hygiene reasons, not accepted in textile bins."),

    ("item_chemical_clothes", "Textile", "NON", "Déchetterie",
     "Vêtements souillés chimiques / peinture | Chemical stained clothing | ملابس ملوثة بمواد كيميائية",
     "Contaminent les autres textiles. Si souillés de produits chimiques → déchetterie zone dangereux. "
     "Contaminates other textiles. Chemical-stained → recycling center hazardous zone."),

    # ── PLASTIQUE — items manquants ───────────────────────────────────────────
    ("item_food_tray", "Plastique", "OUI", "Bac jaune ECOVAL",
     "Barquette alimentaire rincée | Rinsed food tray | صينية طعام مشطوفة",
     "Rincer pour enlever les résidus. Même légèrement souillée reste recyclable. "
     "Code plastique 1 (PET) ou 5 (PP) préférables. "
     "Rinse to remove residue. Even slightly soiled, still recyclable. Code 1 or 5 preferred."),

    ("item_plastic_container", "Plastique", "OUI", "Bac jaune ECOVAL",
     "Bidon plastique propre (eau de Javel, huile) | Clean plastic container bleach oil | جركان بلاستيكي",
     "Vider complètement et rincer 3 fois. Écraser. Le PEHD des bidons est très valorisé. "
     "Éviter les bidons de produits chimiques purs. "
     "Empty completely and rinse 3 times. Crush. HDPE containers highly valued."),

    ("item_plastic_cap", "Plastique", "OUI", "Bac jaune ECOVAL",
     "Bouchon plastique vissé | Screwed plastic cap | سدادة بلاستيكية ملولبة",
     "Le bouchon vissé sur la bouteille écrasée est accepté dans le bac jaune ECOVAL. "
     "Peut aussi être déposé dans les points de collecte sélective. "
     "Cap screwed onto crushed bottle is accepted in the ECOVAL yellow bin."),

    ("item_rigid_packaging", "Plastique", "OUI", "Bac jaune ECOVAL",
     "Emballage plastique rigide propre (code 1, 2, 5) | Clean rigid plastic packaging | تغليف بلاستيكي صلب",
     "PET (1), PEHD (2), PP (5) = les plus recyclables. "
     "Regarder le code triangle en dessous de l'emballage. "
     "PET (1), HDPE (2), PP (5) = most recyclable. Check triangle code on the bottom."),

    ("item_straw", "Plastique", "NON", "Poubelle ordinaire",
     "Paille plastique | Plastic straw | شفاطة بلاستيكية",
     "Trop légère et petite pour les machines de tri. Poubelle ordinaire. "
     "Préférer paille en inox, bambou, papier ou verre réutilisable. "
     "Too light and small for sorting machines. Prefer stainless steel or bamboo straws."),

    ("item_multilayer", "Plastique", "NON", "Poubelle ordinaire",
     "Emballage multicouche (brique de jus, sachets café) | Multi-layer packaging juice carton | تغليف متعدد الطبقات",
     "Mélange plastique + aluminium + carton inséparable mécaniquement. "
     "Pas de filière de recyclage au Maroc actuellement. "
     "Mix of plastic + aluminium + cardboard inseparable. No recycling stream in Morocco yet."),

    # ── PAPIER / CARTON — items manquants ─────────────────────────────────────
    ("item_cereal_box", "Papier", "OUI", "Bac bleu",
     "Boîte carton céréales (Kellogg's) / conserves (Aïcha) | Cardboard box cereals | علبة كرتون",
     "Aplatir. Retirer les sachets plastiques intérieurs (→ bac jaune si recyclable). "
     "Pureté du carton très importante. "
     "Flatten. Remove inner plastic bags. Cardboard purity is very important."),

    ("item_envelope", "Papier", "OUI", "Bac bleu",
     "Enveloppe (avec ou sans fenêtre plastique) | Envelope plastic window | مظروف",
     "La fenêtre plastique est acceptée dans la plupart des collectes. "
     "Retirer les autocollants si possible. "
     "The plastic window is accepted in most collections. Remove stickers if possible."),

    ("item_office_paper", "Papier", "OUI", "Bac bleu",
     "Cahier / Papier bureau (fax, imprimante) | Notebook office paper | دفتر / ورق مكتبي",
     "Retirer les spirales métalliques (→ bac jaune). Les trombones et agrafes sont acceptés. "
     "Ne pas mixer avec papier mouillé. "
     "Remove metal spirals (→ yellow bin). Paper clips and staples are accepted."),

    ("item_directory", "Papier", "OUI", "Bac bleu / Ferracha",
     "Annuaire téléphonique / catalogue | Phone directory catalogue | دليل هاتفي / كتالوج",
     "Papier non couché = très recyclable. La ferracha marocaine rachète ce type de papier. "
     "Dépôt en bac bleu ou point de collecte. "
     "Uncoated paper = very recyclable. The Moroccan ferracha buys this type of paper."),

    ("item_baking_paper", "Papier", "NON", "Poubelle ordinaire",
     "Papier cuisson siliconé | Silicone baking paper | ورق خبز مطلي بالسيليكون",
     "Le revêtement silicone empêche le recyclage. Poubelle ordinaire. "
     "Préférer tapis de cuisson réutilisable en silicone. "
     "Silicone coating prevents recycling. Prefer reusable silicone baking mat."),

    ("item_gift_wrap", "Papier", "NON", "Poubelle ordinaire",
     "Emballage cadeau brillant / pailleté | Shiny glittery gift wrap | ورق تغليف هدايا لامع",
     "Le revêtement plastique ou métallique est non séparable. Poubelle ordinaire. "
     "Préférer papier kraft naturel recyclable. "
     "Plastic or metallic coating inseparable. Prefer recyclable natural kraft paper."),

    ("item_alu_paper", "Papier", "OUI", "Bac jaune ECOVAL (métal)",
     "Papier aluminium / papier cuisson alu | Aluminium foil baking paper | ورق ألمنيوم",
     "Ce n'est pas du papier — c'est de l'aluminium ! "
     "Froisser en boule compacte → bac jaune (métal), pas le bac bleu. "
     "It's not paper — it's aluminium! Crumple into a ball → yellow bin (metal), NOT blue bin."),

    ("item_fastfood_wrap", "Papier", "NON", "Poubelle ordinaire",
     "Emballage fast-food gras (sac McDo, KFC) | Greasy fast-food packaging | تغليف وجبات سريعة دهنية",
     "Imprégné de graisse et souvent mélangé avec plastique. Non recyclable. Poubelle ordinaire. "
     "Soaked in grease and often mixed with plastic. Not recyclable."),

    # ── DÉCHETS DANGEREUX — items manquants ───────────────────────────────────
    ("item_batteries_all", "Dangereux", "COLLECTE SPÉCIALE", "Collecteur ECOVAL",
     "Piles et batteries (toutes tailles) | Batteries all sizes | بطاريات (جميع الأحجام)",
     "Contiennent mercure, cadmium, plomb. Collecteurs ECOVAL dans les supermarchés. "
     "JAMAIS dans la poubelle ordinaire (Loi 28-00). "
     "Contain mercury, cadmium, lead. ECOVAL collectors in supermarkets. NEVER in regular bin."),

    ("item_chemical_products", "Dangereux", "COLLECTE SPÉCIALE", "Déchetterie — dangereux",
     "Produits ménagers chimiques (Javel, déboucheurs) | Chemical household products bleach | منتجات منزلية كيميائية",
     "Ne jamais mélanger (réactions dangereuses). Garder en contenant d'origine. "
     "Déchetterie zone produits dangereux. "
     "Never mix (dangerous reactions). Keep in original container. Hazardous zone."),

    ("item_thermometer", "Dangereux", "COLLECTE SPÉCIALE", "Pharmacie / Déchetterie",
     "Thermomètre à mercure | Mercury thermometer | ميزان حرارة زئبقي",
     "Le mercure est extrêmement toxique. En cas de casse : aérer, évacuer, ne pas aspirer. "
     "Rapporter entier en pharmacie ou déchetterie. "
     "Mercury is extremely toxic. If broken: ventilate, evacuate, do not vacuum."),

    ("item_pesticides", "Dangereux", "COLLECTE SPÉCIALE", "Déchetterie — dangereux",
     "Pesticides / Herbicides / Insecticides | Pesticides herbicides | مبيدات حشرية",
     "Contaminants des eaux souterraines. Déchetterie zone dangereux. "
     "En milieu agricole : programme ECOPHYTO du Maroc. "
     "Groundwater contaminants. In agricultural settings: Morocco's ECOPHYTO programme."),

    ("item_acids", "Dangereux", "COLLECTE SPÉCIALE", "Déchetterie — dangereux",
     "Acides / Bases concentrées | Concentrated acids bases | أحماض / قواعد مركزة",
     "Produits très corrosifs. Conserver en contenant d'origine, ne jamais transférer. "
     "Déchetterie zone dangereux. "
     "Highly corrosive. Store in original container, never transfer to another bottle."),

    ("item_extinguisher", "Dangereux", "COLLECTE SPÉCIALE", "Déchetterie / Professionnel",
     "Extincteur (poudre, CO₂, eau) | Fire extinguisher | طفاية حريق",
     "Certains distributeurs assurent la reprise. Déchetterie ou professionnel agréé. "
     "Ne jamais percer ni jeter à la poubelle. "
     "Some distributors offer take-back. Never pierce or throw in the bin."),
]

# ═══════════════════════════════════════════════════════════════════════════════
# 4. TYPES DE BACS / CONTENEURS
# ═══════════════════════════════════════════════════════════════════════════════

BIN_DOCS = [
    ("bin_yellow", """TYPE DE BAC — Bac Jaune ECOVAL
Matières acceptées : Plastiques (PET, PEHD, PP), métaux (alu, acier), emballages rigides.
Contenu : Bouteilles PET rincées, canettes alu, boîtes conserve acier, barquettes plastique rigide, aérosols vides dégazés.
PAS dans le bac jaune : film plastique souple, polystyrène, verre, papier.
Présent dans : Casablanca, Rabat, Marrakech, Tanger, Agadir, Fès et 20+ villes.
ECOVAL yellow bin accepts: plastics, metals, rigid packaging. Not: soft film, polystyrene, glass.
الحاوية الصفراء ECOVAL: تقبل البلاستيك والمعادن والتغليف الصلب. لا: الغشاء المرن أو الزجاج."""),

    ("bin_green", """TYPE DE BAC — Colonne Verte (verre)
Matières acceptées : Verre creux uniquement (bouteilles, bocaux, flacons).
PAS : Verre brisé, miroir, pyrex, verre trempé, ampoules, céramique.
Rincer avant dépôt. Bouchon métal peut rester.
Présente dans 20 villes marocaines.
Green column: hollow glass only. Not: broken glass, mirror, pyrex, bulbs.
العمود الأخضر: الزجاج المجوف فقط. لا: الزجاج المكسور أو المرايا."""),

    ("bin_blue", """TYPE DE COLLECTE — Ferracha / Papier-Carton au Maroc
IMPORTANT : Le bac bleu pour papier/carton est très peu déployé pour les ménages au Maroc.
C'est la ferracha (récupérateur informel) qui assure l'essentiel du recyclage du papier au Maroc.
Matières recyclables : Journaux, magazines, carton ondulé aplati, boîtes carton, enveloppes, cahiers, briques alimentaires rincées.
PAS recyclable : Ticket de caisse (BPA), carton pizza gras, papier hygiénique, papier aluminium.
Pour recycler son papier : donner à la ferracha ou déposer dans les rares bacs bleus disponibles.
Blue bins are rare in Morocco. The ferracha (informal collector) ensures most paper recycling.
الحاوية الزرقاء نادرة في المغرب. الفراشة (الجامع غير الرسمي) تضمن معظم تدوير الورق."""),

    ("bin_brown", """TYPE DE BAC — Bac Marron / Composteur (organique)
Matières acceptées : Épluchures, marc de café, sachets thé, pain, coquilles d'œuf, feuilles mortes.
PAS : Viande, poisson, produits laitiers, huile, couches.
65% des déchets marocains sont organiques. Décomposition 2-6 mois.
Brown bin: organic waste. Not: meat, fish, dairy, oil, diapers.
الحاوية البنية: النفايات العضوية. لا: اللحم أو الأسماك أو منتجات الألبان."""),

    ("bin_textile", """TYPE DE BAC — Borne Textile / Souk Fripe
Matières acceptées : Vêtements propres (même abîmés), chaussures, linge de maison, sacs tissu.
PAS : Sous-vêtements usagés, vêtements souillés chimiques.
Tradition sadaqa : don à associations (Caritas Maroc, SOS Villages d'Enfants).
Souk fripe (Derb Ghallef, Mellah de Marrakech).
Textile bin: all clothing (even damaged). Not: used underwear, chemically stained clothes.
حاوية النسيج: الملابس والأحذية. لا: الملابس الداخلية المستعملة أو الملوثة."""),

    ("bin_ecoval_deee", """TYPE DE BAC — Éco-point ECOVAL (DEEE — Déchets Électroniques)
Matières acceptées : Smartphones, tablettes, ordinateurs, TV, électroménager, piles Li-ion, chargeurs.
ECOVAL rembourse jusqu'à 500 DH pour les smartphones déposés.
Points dans : Marjane, Carrefour, Cosmos Electro.
Décret 2-14-782 — reprise obligatoire pour les producteurs.
ECOVAL WEEE point: phones, computers, appliances. Up to 500 DH reimbursed.
نقطة ECOVAL للإلكترونيات DEEE: حتى 500 درهم مسترداد. في مرجان وكارفور."""),
]

# ═══════════════════════════════════════════════════════════════════════════════
# 5. LOIS ET RÉGLEMENTATIONS MAROCAINES
# ═══════════════════════════════════════════════════════════════════════════════

LAW_DOCS = [
    ("law_28_00", """LOI MAROCAINE — Loi 28-00 (2006) — en vigueur
Sujet : Gestion et élimination des déchets (BO n°5480)
Principe Pollueur-Payeur. 7 catégories officielles de déchets.
Interdit abandon et brûlage des déchets dans la nature.
Sanctions : jusqu'à 2 000 000 DH + prison selon gravité de l'infraction.
Oblige les communes à élaborer des Plans Directeurs de Gestion des Déchets.
Un avant-projet de révision a été mis en consultation en septembre 2023 pour renforcer les sanctions.
Law 28-00: waste management framework. Fines up to 2,000,000 DH. Mandatory provincial plans. 2023 reform underway.
قانون 28-00: إطار تدبير النفايات. غرامات تصل إلى 2,000,000 درهم. مراجعة جارية (2023)."""),

    ("law_77_15", """LOI MAROCAINE — Loi 77-15 (2016) — "Zéro Mika" — en vigueur
Sujet : Interdiction des sacs plastique depuis le 1er juillet 2016
Interdit fabrication, import et vente de sacs plastique < 60 microns.
Amendes confirmées :
- Détention pour vente/distribution : 2 000 à 500 000 DH
- Détention matières premières pour fabrication : 200 000 à 1 000 000 DH
- Production de sacs plastiques : 10 000 à 100 000 DH
Prison 1 à 3 mois. Seuls sacs certifiés biodégradables (NM ISO 17088) autorisés.
Law 77-15 "Zero Mika": plastic bag ban since July 2016. Confirmed fines: sale 2,000–500,000 DH, raw materials 200,000–1,000,000 DH, production 10,000–100,000 DH.
قانون 77-15 "زيرو ميكا": حظر الأكياس البلاستيكية منذ يوليو 2016. الغرامات المؤكدة: للبيع 2,000–500,000 درهم."""),

    ("law_deee", """LOI MAROCAINE — Déchets Électroniques (DEEE) au Maroc
Les DEEE (téléphones, ordinateurs, TV, appareils électroménagers) sont classés déchets dangereux par la Loi 28-00.
La filière DEEE au Maroc est en cours de structuration.
Les grands distributeurs (surface > 400 m²) ont l'obligation légale de reprendre les appareils usagés.
Objectif national SNRVD 2030 : 20% de recyclage des déchets ménagers.
Principaux opérateurs DEEE : Eco Zalagh (0661 51 47 11 — ecozalagh.com), programme ECOVAL.
WEEE in Morocco under Law 28-00 (hazardous waste). Large retailers must take back. Eco Zalagh: 0661 51 47 11.
DEEE في المغرب بموجب القانون 28-00 (نفايات خطرة). Eco Zalagh: 0661 51 47 11."""),

    ("law_sngd", """STRATÉGIE NATIONALE — SNRVD + PNVDM 2023-2034 (informations vérifiées)
SNRVD (Stratégie Nationale de Réduction et de Valorisation des Déchets) adoptée en 2019.
PNVDM (Programme National de Valorisation des Déchets Ménagers) lancé en septembre 2023.
Budget PNVDM : 21,14 milliards DH pour réhabiliter 230 décharges non contrôlées d'ici 2034.
Soutien Banque Mondiale : 250 millions USD approuvés en novembre 2024.
Objectifs 2030 : 20% recyclage ménager, 70% recyclage plastique, 25% recyclage industriel, 25 000 emplois.
SNRVD 2019 + PNVDM 2023-2034. Budget: 21.14 billion DH. World Bank: 250M USD (2024). Targets: 20% recycling by 2030.
SNRVD 2019 + PNVDM 2023-2034. ميزانية: 21.14 مليار درهم. هدف 2030: 20٪ إعادة تدوير منزلي."""),

    ("law_paris", """ACCORD INTERNATIONAL — Accord de Paris (2016)
Le Maroc s'engage à réduire ses GES de 42% d'ici 2030 (NDC officielle).
Les déchets représentent environ 4,7% des émissions nationales marocaines.
Cible ODD 12.5 : réduire de 50% la génération de déchets d'ici 2030 par prévention, réutilisation et recyclage.
Paris Agreement: Morocco commits to 42% GHG reduction by 2030 (official NDC). Waste = ~4.7% of emissions.
اتفاق باريس: المغرب يلتزم بخفض 42٪ من الغازات الدفيئة بحلول 2030. النفايات تمثل ~4.7٪ من الانبعاثات."""),

    ("law_circ_economy", """PROGRAMME — Feuille de Route Économie Circulaire 2030 — Programme SWITCH-Med UE-Maroc
Adoptée en 2023 avec appui de l'Union Européenne (programme SWITCH-Med).
4 axes : Production Propre, Consommation Responsable, Gestion Optimisée des Déchets, Innovation & Numérique.
Introduction d'un système de consigne pour bouteilles verre/PET envisagée (pilote Casa et Rabat).
EcoScan s'inscrit dans l'axe Innovation & Numérique.
Circular economy roadmap 2030 with EU SWITCH-Med support. Glass/PET deposit scheme planned.
خارطة طريق الاقتصاد الدائري 2030 بدعم أوروبي SWITCH-Med. نظام وديعة الزجاجات مُخطَّط له."""),

    ("law_99_12", """LOI MAROCAINE — Loi 99-12 — Dahir n°1-14-09 (2014)
Sujet : Charte Nationale de l'Environnement et du Développement Durable
Consacre le droit constitutionnel à un environnement sain.
Principes : précaution, prévention, pollueur-payeur, participation citoyenne.
Base juridique de toutes les politiques environnementales marocaines.
Law 99-12 (Dahir 1-14-09, 2014): National Environment Charter. Constitutional right to healthy environment.
قانون 99-12 (ظهير 1-14-09، 2014): الميثاق الوطني للبيئة. الحق الدستوري في بيئة سليمة."""),

    ("law_bale", """CONVENTION INTERNATIONALE — Convention de Bâle (ratifiée par le Maroc en 1995 — Décret 2-95-540)
Sujet : Contrôle des mouvements transfrontaliers de déchets dangereux
Interdit l'exportation de déchets dangereux vers des pays non-OCDE sans preuve de traitement conforme.
Convention de Bamako (1991) renforce cette protection spécifiquement pour l'Afrique.
Violation passible de sanctions pénales graves.
Basel Convention (ratified 1995, Decree 2-95-540): prohibits hazardous waste export to non-OECD countries.
اتفاقية بازل (مصادق عليها 1995، مرسوم 2-95-540): تحظر تصدير النفايات الخطرة لدول غير OCDE."""),

    ("law_49_17", """LOI MAROCAINE — Loi 49-17 (2022) — NOUVELLE LOI CONFIRMÉE
Dahir n°1-20-78 publié au Bulletin Officiel n°7140 du 3 novembre 2022.
Sujet : Évaluation Environnementale (remplace l'ancienne loi sur les études d'impact).
Crée l'Évaluation Environnementale Stratégique (EES) pour toutes les politiques publiques et plans.
Instaure une procédure simplifiée "Notice d'impact" pour les petits projets.
Crée une Commission Nationale d'Évaluation Environnementale pour les projets d'intérêt national.
Law 49-17 (BO n°7140, Nov 2022): new environmental assessment law. Creates Strategic Environmental Assessment (SEA). Replaces former EIA law.
قانون 49-17 (الجريدة الرسمية 7140، نوفمبر 2022): قانون التقييم البيئي الجديد. يُنشئ التقييم البيئي الاستراتيجي."""),
]

# ═══════════════════════════════════════════════════════════════════════════════
# 6. IMPACTS ENVIRONNEMENTAUX
# ═══════════════════════════════════════════════════════════════════════════════

IMPACT_DOCS = [
    ("impact_alu", """IMPACT ENVIRONNEMENTAL — Aluminium / Métal
Bénéfice : 8 kg CO₂ économisés par kg d'alu recyclé. Économie de 95% d'énergie.
Temps décomposition si non recyclé : 2 700 ans.
L'aluminium est recyclable à l'infini sans perte de qualité.
Aluminum: saves 8 kg CO₂/kg, 95% energy saved. Decomposes in 2,700 years.
الألومنيوم: يوفر 8 كجم CO₂ لكل كجم. يتحلل خلال 2700 سنة."""),

    ("impact_glass", """IMPACT ENVIRONNEMENTAL — Verre
Bénéfice : 300 kg CO₂ économisés par tonne recyclée.
Temps décomposition : 4 000 ans.
Le verre est recyclable à 100% indéfiniment.
Glass: saves 300 kg CO₂/tonne. Decomposes in 4,000 years. 100% recyclable indefinitely.
الزجاج: يوفر 300 كجم CO₂ لكل طن. يتحلل خلال 4000 سنة."""),

    ("impact_plastic", """IMPACT ENVIRONNEMENTAL — Plastique PET
Bénéfice : 1,5 kg CO₂ économisés par kg recyclé.
Temps décomposition : 500 ans.
Les sacs plastiques polluent les océans — interdits au Maroc depuis 2016.
PET plastic: saves 1.5 kg CO₂/kg. Decomposes in 500 years.
بلاستيك PET: يتحلل خلال 500 سنة. الأكياس تلوّث المحيطات."""),

    ("impact_paper", """IMPACT ENVIRONNEMENTAL — Papier / Carton
Bénéfice : 17 arbres sauvés par tonne de papier recyclé.
Temps décomposition : 3 à 6 mois.
Paper/Cardboard: saves 17 trees per tonne. Decomposes in 3-6 months.
الورق والكرتون: يُنقذ 17 شجرة لكل طن. يتحلل في 3-6 أشهر."""),

    ("impact_organic", """IMPACT ENVIRONNEMENTAL — Déchets organiques
65% des déchets marocains sont organiques.
Temps décomposition : 2 à 6 mois au composteur.
Le compost améliore la qualité des sols et réduit les besoins en engrais chimiques.
Organic: 65% of Moroccan waste. Decomposes 2-6 months. Compost improves soil.
العضوي: 65٪ من النفايات المغربية. يتحلل في 2-6 أشهر بالسماد."""),

    ("impact_electronics", """IMPACT ENVIRONNEMENTAL — Électronique / DEEE
Bénéfice : 70 kg CO₂ économisés par ordinateur recyclé.
Temps décomposition : plus de 1 000 ans.
1 téléphone contient or, argent, cuivre, lithium — métaux précieux récupérables.
Electronics: saves 70 kg CO₂/computer. Decomposes > 1,000 years. Contains precious metals.
الإلكترونيات: يوفر 70 كجم CO₂ لكل حاسوب. يتحلل أكثر من 1000 سنة."""),

    ("impact_textile", """IMPACT ENVIRONNEMENTAL — Textile
Bénéfice : 3 200 litres d'eau économisés par kg de coton recyclé.
Temps décomposition : 200 ans (synthétique).
Un jean nécessite 8 000 L d'eau pour être fabriqué.
Textile: saves 3,200 L water/kg cotton. Synthetic decomposes in 200 years.
المنسوجات: توفر 3200 لتر ماء لكل كجم قطن. البنطلون يحتاج 8000 لتر ماء."""),

    ("impact_dangerous", """IMPACT ENVIRONNEMENTAL — Déchets dangereux
1 litre d'huile usagée pollue 1 million de litres d'eau.
1 pile peut contaminer 1 000 litres d'eau souterraine.
Temps décomposition : Permanent (certains métaux lourds).
Dangerous waste: 1L motor oil pollutes 1 million L water. 1 battery contaminates 1,000 L.
النفايات الخطرة: لتر زيت واحد يلوّث مليون لتر ماء. بطارية واحدة تلوّث 1000 لتر."""),
]


# ═══════════════════════════════════════════════════════════════════════════════
# FONCTIONS UTILITAIRES
# ═══════════════════════════════════════════════════════════════════════════════

def build_item_doc(doc_id, category, status, bin_name, names, tips):
    content = (
        f"GUIDE DE TRI — {category} : {names}\n"
        f"Recyclable / Statut : {status}\n"
        f"Où le déposer : {bin_name}\n"
        f"Conseils de tri : {tips}"
    )
    return Document(page_content=content, metadata={"id": doc_id, "type": "guide_item", "category": category})


def build_generic_doc(doc_id, content, doc_type):
    return Document(page_content=content, metadata={"id": doc_id, "type": doc_type})


def main():
    print("🔄 Initialisation du modèle d'embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True}
    )

    print("🔄 Connexion à ChromaDB...")
    db = Chroma(
        persist_directory=CHROMA_PATH,
        collection_name=COLLECTION,
        embedding_function=embeddings
    )

    docs = []

    # Navigation
    print("📱 Préparation des documents de navigation...")
    for doc_id, content in NAV_DOCS:
        docs.append(build_generic_doc(doc_id, content, "navigation"))

    # Opérateurs / contacts
    print("📞 Préparation des contacts opérateurs...")
    for doc_id, content in OPERATOR_DOCS:
        docs.append(build_generic_doc(doc_id, content, "contact"))

    # Items guide
    print("♻️  Préparation des items du guide de tri...")
    for doc_id, category, status, bin_name, names, tips in GUIDE_ITEMS:
        docs.append(build_item_doc(doc_id, category, status, bin_name, names, tips))

    # Bacs
    print("🗑️  Préparation des types de bacs...")
    for doc_id, content in BIN_DOCS:
        docs.append(build_generic_doc(doc_id, content, "bin"))

    # Lois
    print("⚖️  Préparation des lois marocaines...")
    for doc_id, content in LAW_DOCS:
        docs.append(build_generic_doc(doc_id, content, "law"))

    # Impacts
    print("🌍 Préparation des impacts environnementaux...")
    for doc_id, content in IMPACT_DOCS:
        docs.append(build_generic_doc(doc_id, content, "impact"))

    print(f"\n✅ {len(docs)} documents préparés.")
    print("💾 Ingestion dans ChromaDB...")

    # Supprimer les anciens documents avec les mêmes IDs pour éviter les doublons
    ids = [doc.metadata["id"] for doc in docs]
    try:
        db._collection.delete(ids=ids)
        print(f"   ↳ Anciens documents supprimés (si existants).")
    except Exception:
        pass

    db.add_documents(docs, ids=ids)
    print(f"✅ Ingestion terminée — {len(docs)} documents ajoutés dans ChromaDB.")
    print(f"   Collection : {COLLECTION}")
    print(f"   Chemin     : {CHROMA_PATH}")


if __name__ == "__main__":
    main()
