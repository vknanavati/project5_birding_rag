# ============================================================
# 01_build_knowledge_base.py
#
# PURPOSE: Create the bird knowledge base — one .txt file per
# species. These documents are the foundation of our entire
# RAG system. The better the documents, the better the answers.
# ============================================================

import os  # lets us create folders and file paths

# ── Where to save the bird documents ──────────────────────────
# os.path.join builds a file path that works on any operating system
OUTPUT_DIR = os.path.join("data", "birds")

# Create the folder if it doesn't exist yet
# exist_ok=True means "don't throw an error if it already exists"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════
# THE KNOWLEDGE BASE
#
# Each entry in this dictionary is one bird species.
# Key   = the filename (e.g. "american_robin" → american_robin.txt)
# Value = the full text document for that species
#
# Writing style tips:
#   - Be specific and factual
#   - Include CT and PA specific details where possible
#   - Cover: identification, habitat, behavior, diet, how to attract
#   - The more detail here, the better the RAG answers will be
# ══════════════════════════════════════════════════════════════

BIRDS = {

"american_robin": """
American Robin (Turdus migratorius)

IDENTIFICATION
The American Robin is one of the most recognizable birds in North America. Adults have a distinctive orange-red breast, dark gray to black back and wings, and a yellow bill. The head is darker than the back, nearly black in males. Females are paler overall with a less vivid orange breast. Robins are large thrushes, measuring 9 to 11 inches in length with a wingspan of 12 to 16 inches. In flight, look for the warm orange underparts and white patches at the corners of the tail.

HABITAT AND RANGE IN CT AND PA
American Robins are year-round residents across Connecticut and Pennsylvania, though their numbers swell dramatically in spring as northern migrants pass through. They thrive in a wide variety of habitats including lawns, parks, gardens, forest edges, and suburban backyards. In winter, robins often gather in large flocks and move into areas with abundant berry-producing trees. In spring and summer they spread out into territories across nearly every neighborhood.

BEHAVIOR
Robins are famous for their behavior of running across lawns, stopping suddenly, tilting their head, and pulling earthworms from the ground. Contrary to popular belief, they locate worms primarily by sight rather than sound. They are early risers and among the first birds to sing at dawn. Males sing a rich, cheerful caroling song often described as "cheerily, cheer-up, cheerio." Robins typically raise two to three broods per season.

DIET
American Robins eat a varied diet that shifts with the seasons. In spring and summer they focus heavily on earthworms and other invertebrates including beetles, grasshoppers, and caterpillars. In fall and winter they shift to fruits and berries, including holly berries, crabapples, juniper berries, and dogwood fruits. This seasonal fruit diet is why robins sometimes appear in backyards in winter even in cold weather.

HOW TO ATTRACT TO YOUR YARD
Robins do not typically visit seed feeders. The best ways to attract them are: maintain a lawn where they can hunt earthworms, plant native berry-producing trees and shrubs such as holly, dogwood, serviceberry, and crabapple, and provide a birdbath with fresh water since robins are strongly attracted to water for drinking and bathing. During dry spells, a lawn sprinkler will often bring robins running.

SIMILAR SPECIES
The Varied Thrush resembles the Robin but has an orange eyebrow stripe and wing bars. It is rare on the East Coast. Juvenile Robins before their first molt have spotted breasts similar to other thrushes but retain the Robin's overall size and structure.
""",

"black_capped_chickadee": """
Black-capped Chickadee (Poecile atricapillus)

IDENTIFICATION
The Black-capped Chickadee is a tiny, acrobatic bird with a distinctive black cap and bib, white cheeks, gray back and wings, and buffy-white underparts. It measures just 4.7 to 5.9 inches in length. The cap is a crisp, solid black that extends to just below the eye. The white cheek patch is bright and clean. Wings show thin white edging on the feathers. Males and females look identical. Their small size and round shape combined with the bold black-and-white head pattern make them unmistakable.

HABITAT AND RANGE IN CT AND PA
Black-capped Chickadees are permanent year-round residents throughout Connecticut and Pennsylvania. They inhabit deciduous and mixed forests, forest edges, parks, and suburban backyards with mature trees. They are one of the most common feeder visitors across both states in all seasons. In winter they often join mixed flocks with nuthatches, Downy Woodpeckers, and creepers.

BEHAVIOR
Chickadees are extraordinarily bold and curious. With patience they can be trained to eat from your hand. They are acrobatic feeders, often hanging upside down from branch tips to reach food. They are known for caching food — hiding individual seeds in bark crevices and under leaves — and can remember thousands of cache locations. Their signature call is a clear, whistled "fee-bee" (two notes, the second lower). Their chickadee-dee-dee alarm call varies in intensity based on the level of threat: more "dees" means greater danger.

DIET
Black-capped Chickadees eat insects, seeds, and berries. In warm months, insects and spiders make up the majority of their diet. In winter they rely heavily on seeds, particularly sunflower seeds, and animal fat from suet. They are one of the few small birds capable of surviving very cold winters due to a combination of food caching, nocturnal hypothermia (lowering body temperature overnight to conserve energy), and roosting in tree cavities.

HOW TO ATTRACT TO YOUR YARD
Chickadees are among the easiest birds to attract. They readily visit feeders offering black oil sunflower seeds, hulled sunflower chips, and suet. They prefer tube feeders and platform feeders. Providing a nest box with a 1.125-inch entrance hole in early spring may attract a nesting pair. Planting native trees like oaks, birches, and willows supports the caterpillar populations that chickadees depend on for raising young.

SIMILAR SPECIES
In southern Connecticut and Pennsylvania, the Carolina Chickadee is nearly identical to the Black-capped. The best field marks are the Carolina's smaller size, slightly cleaner wing panel, and different song — a four-note "fee-bee-fee-bay" rather than the two-note "fee-bee." Where the two species overlap they sometimes hybridize, making identification very difficult.
""",

"carolina_wren": """
Carolina Wren (Thryothorus ludovicianus)

IDENTIFICATION
The Carolina Wren is a small but surprisingly loud bird with rich, warm reddish-brown upperparts, buffy-orange underparts, and a bold white eyebrow stripe. It measures 4.9 to 5.5 inches long. The bill is long and slightly downcurved, suited for probing into bark and leaf litter. The tail is often held cocked upward. Males and females look alike. Despite its small size, the Carolina Wren has one of the loudest voices relative to body size of any North American bird.

HABITAT AND RANGE IN CT AND PA
Carolina Wrens are year-round residents in Pennsylvania and southern Connecticut, though their range has been expanding northward in recent decades as winters have become milder. They prefer dense, brushy habitats — thickets, overgrown hedgerows, tangles of vines, brushy forest edges, and suburban gardens with dense shrubs. They rarely venture far from dense cover and are often heard long before they are seen.

BEHAVIOR
Carolina Wrens are curious and bold but secretive, spending much of their time foraging low in dense vegetation. They investigate holes, gaps, and crevices constantly. Males sing year-round, even in winter — an unusual trait among songbirds. The song is a loud, ringing "teakettle-teakettle-teakettle" that can be heard from a considerable distance. They are non-migratory and stay paired year-round, which is uncommon among small songbirds. Pairs maintain territories together through all seasons.

DIET
Carolina Wrens are primarily insectivores. They eat beetles, caterpillars, grasshoppers, crickets, spiders, and small lizards. They forage by probing bark, flipping leaves, and investigating crevices. In winter when insects are scarce they supplement their diet with seeds and berries, and will visit suet feeders regularly.

HOW TO ATTRACT TO YOUR YARD
Provide suet feeders in fall and winter — Carolina Wrens are reliable suet visitors. They will also eat peanut pieces and mealworms from platform feeders placed low to the ground. The most important factor is dense cover: planting native shrubs like spicebush, viburnum, and elderberry creates the brushy habitat they require. They will sometimes nest in unusual locations including hanging flower baskets, old boots left on a porch, and open-fronted nest boxes.

SIMILAR SPECIES
The House Wren is smaller, plainer brown, and lacks the bold white eyebrow stripe of the Carolina Wren. The Marsh Wren and Sedge Wren are found in wetland habitats rather than backyards. The Carolina Wren's size, warm coloring, and bold eyebrow stripe make it fairly distinctive.
""",

"downy_woodpecker": """
Downy Woodpecker (Dryobates pubescens)

IDENTIFICATION
The Downy Woodpecker is North America's smallest woodpecker, measuring just 5.5 to 6.7 inches in length. It has a bold black-and-white pattern: white underparts, black wings with white spots, a white stripe down the back, and a black-and-white striped head. Males have a small red patch on the back of the head; females do not. The bill is short and stubby relative to other woodpeckers — a key field mark. The outer tail feathers are white with small black spots.

HABITAT AND RANGE IN CT AND PA
Downy Woodpeckers are permanent year-round residents throughout Connecticut and Pennsylvania. They are found in nearly every wooded habitat including deciduous forests, forest edges, orchards, parks, and suburban backyards with mature trees. They are one of the most frequent visitors to backyard feeders in both states across all seasons.

BEHAVIOR
Downies forage by hitching up tree trunks and branches, probing bark with their bills to find insects hidden inside. They excavate nest cavities in dead wood each spring. Outside of the breeding season they often join mixed flocks with chickadees and nuthatches. Males and females partition their foraging habitat slightly — males tend to forage on smaller branches and weed stems while females work the main trunk. Their call is a soft "pik" and they produce a descending whinny call. They drum on resonant wood surfaces to communicate.

DIET
Downy Woodpeckers eat insects and larvae hidden under bark, including beetle larvae, ants, and caterpillars. They also eat berries, seeds, and acorns. At feeders they are strongly attracted to suet, which provides the concentrated fat they need particularly in winter. They also eat black oil sunflower seeds and peanut butter.

HOW TO ATTRACT TO YOUR YARD
Suet is the single most effective food for attracting Downy Woodpeckers. Use a cage-style suet feeder hung from a tree or feeder pole. They also visit tube feeders for sunflower seeds and peanut pieces. Leaving dead trees or large dead branches standing on your property provides foraging habitat and potential nest sites. They will use nest boxes with a 1.25-inch entrance hole packed with wood shavings.

SIMILAR SPECIES
The Hairy Woodpecker is nearly identical to the Downy but noticeably larger (9 to 11 inches) with a much longer, heavier bill. The Hairy's outer tail feathers are pure white without spots. Bill length relative to head size is the most reliable field mark when both species aren't present for direct comparison.
""",

"american_goldfinch": """
American Goldfinch (Spinus tristis)

IDENTIFICATION
The American Goldfinch is one of the most brilliantly colored backyard birds. In summer, breeding males are vivid lemon-yellow with jet black wings showing two white wing bars, a black forehead patch, and a white rump visible in flight. Females and winter males are much duller — olive-yellow to grayish-yellow with darker wings and the same white wing bars. The bill is small, conical, and pinkish-orange. Goldfinches measure 4.3 to 5.1 inches in length. They have a distinctive bouncy, undulating flight pattern.

HABITAT AND RANGE IN CT AND PA
American Goldfinches are year-round residents throughout Connecticut and Pennsylvania, though they are partially migratory and numbers increase in summer. They prefer open habitats including meadows, fields, roadsides, and the edges of forests. They are extremely common at backyard feeders, particularly during fall and winter when they congregate in large flocks.

BEHAVIOR
Goldfinches are highly social and often travel and feed in flocks. They are strict vegetarians — unusual among songbirds — feeding almost exclusively on seeds year-round. They time their breeding season to coincide with the peak abundance of thistle and other seed-bearing plants in midsummer, making them one of the latest-nesting songbirds in North America. Their song is a series of musical twitters and their flight call is a distinctive "po-ta-to-chip" sound heard as they fly overhead.

DIET
American Goldfinches eat seeds almost exclusively. Their favorites include thistle (nyjer) seeds, sunflower seeds, and the seeds of wildflowers like coneflowers, cosmos, and black-eyed Susans. They feed their nestlings regurgitated seeds rather than insects, which is unusual among songbirds. They are often seen clinging to seed heads of tall wildflowers to extract seeds directly.

HOW TO ATTRACT TO YOUR YARD
Nyjer (thistle) seed in a tube feeder with small ports is the single best way to attract goldfinches. They also readily eat hulled sunflower chips. Planting native wildflowers that produce seed heads — coneflowers (Echinacea), black-eyed Susans, and native thistles — provides natural foraging and is extremely attractive to goldfinches in late summer and fall. Allow spent flower heads to remain standing through winter rather than cutting them back.

SIMILAR SPECIES
In winter when goldfinches are dull and yellowish, they can be confused with Pine Siskins (which have streaky brown plumage and yellow wing flashes) and Common Redpolls (which have red foreheads and pink breasts). The goldfinch's unstreaked body and white wing bars help distinguish it.
""",

"northern_cardinal": """
Northern Cardinal (Cardinalis cardinalis)

IDENTIFICATION
The Northern Cardinal is one of the most instantly recognizable birds in North America. Adult males are brilliant red over their entire body with a black mask that covers the face and throat, a prominent pointed crest, and a heavy orange-red conical bill. Females are warm buffy-brown with red tinges on the crest, wings, and tail, the same black mask, and the same heavy orange bill. Juveniles resemble females but have a dark bill. Cardinals measure 8.3 to 9.1 inches in length. The crest is distinctive — no other common backyard bird in the East has this prominent pointed head feather.

HABITAT AND RANGE IN CT AND PA
Northern Cardinals are permanent year-round residents throughout Connecticut and Pennsylvania. They are among the most common backyard birds in both states. They prefer forest edges, thickets, hedgerows, overgrown gardens, and suburban backyards with dense shrubs for cover. They are not migratory and pairs often maintain territories year-round near reliable food sources.

BEHAVIOR
Cardinals are primarily ground and low-shrub foragers. Males are territorial and will aggressively defend feeding areas from other males, sometimes fighting their own reflection in windows. Both males and females sing — female songbirds that sing are relatively rare in North America. The song is a series of loud, clear whistles often described as "cheer cheer cheer" or "what-cheer what-cheer." Cardinals are among the first birds to visit feeders at dawn and among the last to leave at dusk.

DIET
Northern Cardinals eat primarily seeds, grains, and fruits. Their heavy bill is well adapted for cracking open large seeds. Sunflower seeds are a staple, particularly black oil sunflower seeds and safflower seeds. They also eat wild berries, corn, and in summer, insects and caterpillars which they feed to their young.

HOW TO ATTRACT TO YOUR YARD
Cardinals are reliably attracted to platform feeders and hopper feeders stocked with black oil sunflower seeds or safflower seeds. Safflower is particularly useful because squirrels and grackles tend to avoid it while cardinals love it. They prefer to feed at or near ground level, so low platform feeders work better than high tube feeders. Dense shrubs like viburnums, hollies, and spicebush provide the cover they need and feel safe approaching feeders nearby.

SIMILAR SPECIES
No other eastern bird resembles the male Cardinal. The female can be confused with other brownish birds but the combination of crest, orange bill, and red tinges is distinctive. The Pyrrhuloxia, a southwestern relative, does not occur in the East.
""",

"house_finch": """
House Finch (Haemorhous mexicanus)

IDENTIFICATION
House Finches are small, chunky finches measuring 5 to 6 inches in length. Adult males have a raspberry-red to orange-red wash on the head, breast, and rump, with brown-streaked back and wings, and a streaked belly. The intensity of the red coloration varies considerably between individuals and is determined by diet — males with access to better food sources during molt develop brighter red color. Females are plain brown with heavy streaking throughout, no distinctive head markings, and a slightly curved bill. The curved culmen (top edge of the bill) is a useful field mark separating House Finches from Purple Finches.

HABITAT AND RANGE IN CT AND PA
House Finches are year-round residents throughout Connecticut and Pennsylvania. They thrive in urban and suburban environments — shopping centers, farms, forest edges, and backyard feeders. They are an introduced species on the East Coast, originally released on Long Island in 1940 after being illegally sold as cage birds, and have since spread across the entire eastern United States.

BEHAVIOR
House Finches are highly social and gregarious, feeding in flocks year-round. They are vocal birds with a long, rambling, musical song that males sing frequently. They nest in a wide variety of locations including hanging planters, wreaths, and dense shrubs near houses. Females do all the incubation while males bring food. They are susceptible to a bacterial eye disease called Mycoplasmal conjunctivitis that causes swollen, crusty eyes — affected birds should be noted and feeders cleaned regularly to prevent spread.

DIET
House Finches are primarily seed eaters. They eat sunflower seeds, nyjer seeds, millet, and the seeds of many wild plants. They also eat berries and some flower buds in spring. At feeders they are one of the most common visitors to tube feeders and platform feeders.

HOW TO ATTRACT TO YOUR YARD
House Finches readily visit tube feeders filled with black oil sunflower seeds or nyjer seed. They are less picky than many finches and will use almost any feeder style. To minimize disease transmission, clean feeders regularly with a dilute bleach solution and remove any birds showing eye symptoms from your feeder records.

SIMILAR SPECIES
Purple Finches are very similar. Key differences: male Purple Finch is raspberry-red suffused across the whole head and back without streaking on the breast sides; male House Finch has a clearly streaked belly and brown back. Female Purple Finch has a bold white eyebrow stripe; female House Finch does not. The Purple Finch's bill is less curved than the House Finch's.
""",

"purple_finch": """
Purple Finch (Haemorhous purpureus)

IDENTIFICATION
Despite the name, Purple Finches are not purple — males are raspberry-red, as if dipped in raspberry juice, with the color suffused over the head, back, and breast without the crisp streaking seen on House Finches. The back has reddish tones mixed into the brown. Females are heavily streaked brown and white with a bold white eyebrow stripe and a whitish cheek patch outlined in dark brown — a strong pattern absent on the female House Finch. Purple Finches measure 4.7 to 6.3 inches in length. The bill is slightly less curved than the House Finch's.

HABITAT AND RANGE IN CT AND PA
Purple Finches are primarily winter visitors in Connecticut and Pennsylvania, arriving from their boreal breeding grounds in October and departing by April. Some breed in northern Connecticut and at higher elevations in Pennsylvania. They prefer coniferous and mixed forests and visit backyard feeders during winter irruptions when food is scarce to the north. Their numbers vary considerably from year to year — some winters bring large flocks, others very few.

BEHAVIOR
Purple Finches travel in flocks in winter and associate with House Finches and American Goldfinches at feeders. Males sing a rich, fast warble — longer and more complex than the House Finch's song. They are somewhat less bold than House Finches and may be displaced from feeders by more aggressive species. In their breeding habitat they forage in the treetops for buds, seeds, and berries.

DIET
Purple Finches eat seeds, berries, and buds. At feeders they prefer sunflower seeds. In the wild they eat the seeds of ash, tulip poplar, and elms, as well as wild berries including elderberries.

HOW TO ATTRACT TO YOUR YARD
Black oil sunflower seeds in tube feeders or platform feeders are the best option. Since Purple Finches visit primarily in winter, keeping feeders stocked from October through April maximizes your chances during irruption years. Planting berry-producing native shrubs like elderberry, dogwood, and viburnum also attracts them.

SIMILAR SPECIES
See House Finch account for detailed comparison. The Cassin's Finch, a western species, does not occur in the East. In winter, female Purple Finches can also be confused with female Rose-breasted Grosbeaks, which are larger with a heavier bill.
""",

"white_breasted_nuthatch": """
White-breasted Nuthatch (Sitta carolinensis)

IDENTIFICATION
The White-breasted Nuthatch is a compact, short-tailed bird with a long, straight, pointed bill and a habit of climbing headfirst down tree trunks — a behavior no other common backyard bird performs. Adults have a blue-gray back, black cap (gray in females), white face and underparts, and chestnut coloring on the lower belly and under the tail. The tail appears very short in the field. They measure 5.1 to 5.5 inches in length. The headfirst descending posture is the single most reliable field identification clue.

HABITAT AND RANGE IN CT AND PA
White-breasted Nuthatches are permanent year-round residents throughout Connecticut and Pennsylvania. They prefer mature deciduous and mixed forests with large trees, as well as wooded suburban areas, orchards, and parks. They require large-diameter trees for both foraging and nesting. They are non-migratory and pairs often remain together on the same territory year-round.

BEHAVIOR
Nuthatches forage by walking in all directions on tree bark — up, down, and sideways — probing crevices for hidden insects and larvae. Their headfirst descent down trunks gives them a unique view of crevices that upward-climbing birds like woodpeckers miss. They cache food by wedging seeds into bark furrows and covering them with bark flakes or lichen. Their call is a distinctive nasal "yank-yank" or "whi-whi-whi" that carries well through the woods. They often join mixed foraging flocks with chickadees and Downy Woodpeckers in winter.

DIET
White-breasted Nuthatches eat insects, larvae, and seeds. In warmer months they focus on insects and spiders gleaned from bark. In winter they rely more heavily on seeds and nuts including acorns, which they wedge into bark and hammer open (hence the name "nuthatch" — nut hacker). At feeders they are strongly attracted to sunflower seeds, suet, and peanuts.

HOW TO ATTRACT TO YOUR YARD
Sunflower seeds and suet are the best feeder foods for nuthatches. They prefer to take one seed at a time and cache it nearby, so they visit feeders frequently in quick trips. Providing mature trees on your property — especially oaks — is the most important habitat factor. They will use nest boxes with a 1.25 to 1.375-inch entrance hole, preferably mounted on a tree rather than a pole.

SIMILAR SPECIES
The Red-breasted Nuthatch is smaller with a rusty-orange breast and a bold white eyebrow stripe, and is primarily a winter visitor to the region. The Brown-headed Nuthatch and Pygmy Nuthatch do not occur in the Northeast.
""",

"tufted_titmouse": """
Tufted Titmouse (Baeolophus bicolor)

IDENTIFICATION
The Tufted Titmouse is a small, elegant bird with soft gray upperparts, white underparts, a rusty-orange wash on the flanks, a black patch just above the bill, and a prominent gray crest. The large dark eyes give it an alert, almost surprised expression. It measures 5.5 to 6.3 inches in length. Males and females look identical. The pointed crest is the most distinctive feature — no other small gray backyard bird in the East has this crest.

HABITAT AND RANGE IN CT AND PA
Tufted Titmice are permanent year-round residents throughout Connecticut and Pennsylvania. They have expanded their range northward significantly over the past several decades. They prefer deciduous and mixed forests with large trees, particularly oaks and beeches, and are common in suburban and urban areas with mature tree cover. They rarely venture far from trees.

BEHAVIOR
Titmice are bold, active birds and frequent feeder visitors. Like chickadees they are acrobatic foragers, often hanging from branch tips. They are known for their loud, clear whistled song — a repeated "peter-peter-peter" that is one of the first bird songs many people learn. They frequently join mixed foraging flocks with chickadees and nuthatches in winter. A distinctive behavior is lining their nest with soft animal fur — they have been observed plucking fur directly from live animals including dogs, cats, squirrels, and even people sitting outdoors.

DIET
Tufted Titmice eat insects, seeds, nuts, and berries. In summer, insects and caterpillars make up a large part of their diet. In fall and winter they focus on seeds and cache food for later retrieval. They prefer large seeds and will select the largest sunflower seeds available at a feeder. Beech nuts and acorns are important natural foods.

HOW TO ATTRACT TO YOUR YARD
Black oil sunflower seeds are highly attractive to titmice. They also eat suet, peanuts, and mealworms. They readily visit tube feeders, platform feeders, and suet cages. Like chickadees, they can be trained to eat from the hand with patience. Nest boxes with a 1.25-inch entrance hole attract nesting pairs in spring.

SIMILAR SPECIES
No other small eastern backyard bird has both a gray crest and the black forehead patch. Chickadees lack the crest. The titmouse's crest alone makes it essentially unmistakable in Connecticut and Pennsylvania backyards.
""",

"dark_eyed_junco": """
Dark-eyed Junco (Junco hyemalis)

IDENTIFICATION
Dark-eyed Juncos are medium-small sparrows with a distinctive pattern: dark gray to slate-gray hood and upperparts, white belly, and striking white outer tail feathers that flash in flight. The bill is small and pinkish. In the East, the slate-colored form is the standard: males are a clean dark gray above, females are browner-gray. They measure 5.5 to 6.3 inches. The white outer tail feathers flashing as they fly is one of the most reliable field marks and is visible from a considerable distance.

HABITAT AND RANGE IN CT AND PA
Dark-eyed Juncos are winter visitors in Connecticut and Pennsylvania, arriving from their boreal and montane breeding grounds in October and departing by April. They breed at higher elevations in northern Pennsylvania. In winter they are one of the most abundant birds at feeders across both states, often appearing in large flocks. They favor areas with dense low cover near open ground.

BEHAVIOR
Juncos are ground foragers. They scratch through leaf litter and snow to find seeds, making a distinctive scratching sound. They often feed below feeders, picking up seeds dropped by other birds. Flocks have a clear dominance hierarchy. When alarmed they flush simultaneously, flashing their white tail feathers as a group — this startling effect may confuse predators. Their call is a sharp metallic "tick." They roost communally in dense conifers on cold nights.

DIET
Dark-eyed Juncos eat primarily seeds in winter, including millet, sunflower chips, and grass seeds. They also eat some insects and berries when available. They are classic "snowbirds" — their arrival in the yard is often seen as a sign that winter weather is coming.

HOW TO ATTRACT TO YOUR YARD
Juncos prefer to feed on the ground or on low platform feeders. Scatter white proso millet on the ground or on a ground-level platform feeder below your main feeders. They will also eat sunflower chips and cracked corn. Dense shrubs nearby provide essential escape cover. They are one of the easiest winter birds to attract — simply keeping a tidy brush pile near your feeding area is often enough.

SIMILAR SPECIES
No other common eastern sparrow has the combination of dark hood and white outer tail feathers. The Eastern Towhee is larger and has a more complex pattern. Juncos are essentially unmistakable once learned.
""",

"song_sparrow": """
Song Sparrow (Melospiza melodia)

IDENTIFICATION
Song Sparrows are medium-sized sparrows with heavily streaked brown and white plumage, a long rounded tail, and a distinctive large dark spot in the center of the breast. The head shows a gray central crown stripe, brown lateral stripes, a gray eyebrow, and brown ear patch. They measure 4.7 to 6.7 inches. The combination of heavy streaking, central breast spot, and pumping tail in flight are the key identification marks. Song Sparrows are highly variable across their range — eastern birds tend to be richly colored with warm brown tones.

HABITAT AND RANGE IN CT AND PA
Song Sparrows are year-round residents in Connecticut and Pennsylvania, with numbers augmented by migrants in spring and fall. They are found in a wide variety of brushy habitats including marsh edges, old fields, hedgerows, forest edges, stream corridors, and suburban gardens with dense low cover. They are one of the most widespread and adaptable sparrows in North America.

BEHAVIOR
Song Sparrows are persistent, energetic singers. Males sing from exposed perches throughout the year, even in winter on mild days. The song is a complex series of notes beginning with two or three clear notes followed by a buzzy trill — often described as "maids-maids-maids-put-on-your-teakettle-ettle-ettle." Each male has multiple song variations. They pump their tail characteristically in flight. They are generally secretive and stay low in vegetation, but males sing conspicuously from exposed perches.

DIET
Song Sparrows eat seeds, insects, and small aquatic invertebrates. In summer insects make up a substantial portion of the diet, particularly for feeding young. In winter they eat primarily seeds including weed seeds, grass seeds, and grains. They forage by scratching through leaf litter and on the ground.

HOW TO ATTRACT TO YOUR YARD
Song Sparrows visit feeders that offer white proso millet, sunflower chips, and cracked corn, preferably on ground-level platforms. More importantly they are attracted by habitat — dense brush piles, overgrown hedgerows, and shrubby cover near the ground. A water source is also highly attractive. Allowing a section of your yard to grow somewhat wild is one of the best ways to support Song Sparrows.

SIMILAR SPECIES
Song Sparrows can be confused with other streaked sparrows including Lincoln's Sparrow (finer streaking, buffy wash on breast), Savannah Sparrow (shorter tail, yellow eyebrow in some), and Fox Sparrow (larger, richer reddish-brown). The central breast spot is the most reliable quick field mark.
""",

"mourning_dove": """
Mourning Dove (Zenaida macroura)

IDENTIFICATION
Mourning Doves are slender, long-tailed doves with soft grayish-brown plumage, a small rounded head, black spots on the wings, iridescent pink and green on the neck, and a long pointed tail with white outer edges visible in flight. The underparts are a soft pinkish-buff. The eye is dark with a pale blue orbital ring. They measure 9 to 13.4 inches in length — noticeably larger than a Robin's width but sleeker. In flight, the wings produce a distinctive whistling sound.

HABITAT AND RANGE IN CT AND PA
Mourning Doves are year-round residents throughout Connecticut and Pennsylvania, with populations supplemented by migrants in fall. They are found in open habitats including agricultural fields, grasslands, roadsides, and suburban and urban areas. They are among the most abundant birds in both states and one of the most frequently hunted game birds in North America.

BEHAVIOR
Mourning Doves are ground foragers, walking slowly while pecking for seeds. They typically feed in open areas and fly to trees or wires when disturbed. They are often seen in pairs or small groups outside breeding season. The song is a mournful, hollow cooing — "coo-OO-oo-oo-oo" — which gives the species its name and is sometimes mistaken for an owl. They drink by submerging their bill and sucking water continuously, unlike most birds that scoop and tilt their head back.

DIET
Mourning Doves eat almost exclusively seeds — up to 20% of their body weight per day. They favor the seeds of grasses, weeds, grains, and agricultural crops. They swallow seeds whole and grind them in a muscular gizzard. They occasionally eat snails for calcium.

HOW TO ATTRACT TO YOUR YARD
Mourning Doves prefer to feed on the ground or on large platform feeders. Scattering millet, cracked corn, or sunflower chips on the ground is the easiest way to attract them. They are not well suited for tube feeders due to their size. They are attracted to open areas with some bare ground and nearby perching trees.

SIMILAR SPECIES
The Eurasian Collared-Dove is larger and paler with a black collar on the back of the neck. It has been spreading into the East and may occasionally be seen in Connecticut and Pennsylvania. The Rock Pigeon (common city pigeon) is larger, stockier, and more variable in color. The Common Ground-Dove is much smaller and is not expected in the Northeast.
""",

"american_crow": """
American Crow (Corvus brachyrhynchos)

IDENTIFICATION
American Crows are large, entirely black birds — black bill, black legs, black feet, black eyes — with a fan-shaped tail in flight. They measure 15.8 to 20.9 inches in length with a wingspan of 33 to 39 inches. The plumage has a subtle iridescent gloss in good light. In flight the tail is distinctly rounded (versus the wedge-shaped tail of the Common Raven). The voice is the classic "caw-caw-caw" that most people associate with crows. Males and females look identical.

HABITAT AND RANGE IN CT AND PA
American Crows are year-round residents throughout Connecticut and Pennsylvania. They are found in virtually every habitat including farmland, forests, suburban neighborhoods, parks, and coastal areas. In fall and winter they form massive communal roosts that can number in the thousands or even hundreds of thousands of birds. They are highly intelligent and adaptable, thriving in human-modified landscapes.

BEHAVIOR
Crows are among the most intelligent birds in the world. They use tools, recognize individual human faces, hold grudges, communicate complex information, and engage in play. They are omnivores that exploit every available food source. Family groups (parents plus offspring from previous years) defend territories together and cooperate in raising young — offspring from prior years serve as helpers at the nest. Crows are known to mob owls and hawks aggressively, and following an agitated crow mob often leads to finding a roosting raptor.

DIET
American Crows eat almost anything: insects, earthworms, small vertebrates, eggs and nestlings of other birds, carrion, garbage, corn, seeds, fruits, and human food scraps. Their dietary flexibility is central to their success in human-dominated landscapes.

HOW TO ATTRACT TO YOUR YARD
Crows are attracted to yards with open ground, accessible food, and tall perching trees. Corn — whole or cracked — placed on the ground or a large platform is the best feeder food. They will also eat meat scraps and dog food. Many birders have mixed feelings about encouraging crows due to their predation on other songbird nests.

SIMILAR SPECIES
The Common Raven is larger with a wedge-shaped tail, deeper "gronk" call, and heavier bill. Ravens are rare in southern Connecticut and Pennsylvania but regular in northern PA. The Fish Crow (slightly smaller, nasal "uh-uh" call) occurs in coastal and river valley areas of both states. Voice is usually the best way to distinguish American Crow from Fish Crow.
""",

"blue_jay": """
Blue Jay (Cyanocitta cristata)

IDENTIFICATION
Blue Jays are large, crested birds with brilliant blue upperparts, a white face and underparts, a bold black necklace across the throat, and black barring and white spots on the wings and tail. The blue color is produced by light scattering through modified feather barbs rather than pigment — wet feathers lose the blue iridescence. They measure 9.8 to 11.8 inches in length. The prominent blue crest is raised when alert or excited and flattened when feeding or interacting submissively. Males and females look identical.

HABITAT AND RANGE IN CT AND PA
Blue Jays are year-round residents throughout Connecticut and Pennsylvania. They prefer mixed forests with oaks and beeches, forest edges, and suburban areas with mature trees. Many Blue Jays in the Northeast are partially migratory — some individuals migrate south in fall while others remain year-round. Fall migration can produce impressive counts along ridges and coastlines.

BEHAVIOR
Blue Jays are vocal, bold, and intelligent. They produce a wide variety of calls including the classic harsh "jay-jay" alarm call, a musical "toolool" call, and remarkably accurate imitations of Red-shouldered and Red-tailed Hawk calls — used to alert other jays to hawk presence or possibly to cause other birds to flee feeders. They are important forest seed dispersers, caching tens of thousands of acorns each fall in scattered locations and recovering most but not all of them — forgotten caches contribute substantially to oak forest regeneration.

DIET
Blue Jays are omnivores. Acorns, beech nuts, and seeds form the core of their diet. They also eat insects, small vertebrates, eggs and nestlings of other birds, berries, and corn. At feeders they prefer whole peanuts in the shell, sunflower seeds, and cracked corn. Their large bill allows them to handle food items that smaller birds cannot.

HOW TO ATTRACT TO YOUR YARD
Whole peanuts in the shell placed on a platform feeder are irresistible to Blue Jays. They will often grab multiple peanuts, stuffing them into their throat pouch (gular pouch) before flying off to cache them. Sunflower seeds, cracked corn, and suet also attract them. Planting native oaks is the single best long-term action for supporting Blue Jays.

SIMILAR SPECIES
No other large eastern bird has the Blue Jay's combination of blue crest, black necklace, and white-spotted wings. The Steller's Jay (western) has a darker, blacker head and does not occur in the East.
""",

"european_starling": """
European Starling (Sturnus vulgaris)

IDENTIFICATION
European Starlings are medium-sized, stocky birds with a short tail and a long, pointed yellow bill (in breeding season) or dark bill (in winter). In spring and summer adults have iridescent black plumage with green and purple gloss. In fall and winter they are heavily spotted with white and buff, giving a speckled appearance, and the bill turns dark. They measure 7.9 to 9.1 inches. In flight, the short tail and triangular wing shape are distinctive. Murmurations — the stunning aerial flocking displays of thousands of starlings moving in coordinated waves — are one of nature's most remarkable sights.

HABITAT AND RANGE IN CT AND PA
European Starlings are year-round residents throughout Connecticut and Pennsylvania and virtually everywhere else in North America. They are an introduced species, released in New York City's Central Park in 1890 by a group who wanted to introduce all birds mentioned by Shakespeare. From that introduction they spread across the entire continent. They thrive in agricultural areas, suburban neighborhoods, urban centers, and wherever there are open areas for foraging and structures for nesting.

BEHAVIOR
Starlings are highly social and form enormous flocks, particularly in fall and winter. They are aggressive cavity nesters that compete directly with native hole-nesting birds like bluebirds, Tree Swallows, and Purple Martins for nesting sites. Their vocalizations are complex — they are gifted mimics that incorporate the calls of many other species including Red-tailed Hawk, Killdeer, and even mechanical sounds. They walk rather than hop on the ground, a useful field mark.

DIET
European Starlings are omnivores that eat insects, earthworms, berries, fruits, seeds, and garbage. They are highly effective at foraging in short grass and agricultural fields. They often displace native species at suet feeders.

HOW TO ATTRACT OR DETER
Most native bird enthusiasts actively try to limit starling access to feeders. Using upside-down suet feeders (where birds must cling from below) deters starlings, which prefer to feed right-side-up, while allowing woodpeckers and nuthatches. Safflower seeds are less attractive to starlings than sunflower. Monitoring nest boxes and removing starling nests protects cavity-nesting natives.

SIMILAR SPECIES
In winter spotted plumage, Starlings are distinctive. Spring adults in iridescent black can be confused at a distance with Rusty Blackbirds or Brewer's Blackbirds (both rare in the region), but the Starling's short tail and pointed bill separate it.
""",

"house_sparrow": """
House Sparrow (Passer domesticus)

IDENTIFICATION
House Sparrows are small, stocky sparrows introduced from Europe. Males have a gray crown, chestnut nape, white cheeks, black bib (larger in dominant males), streaked brown back, and gray underparts. Females and juveniles are plain buffy-brown with a pale eyebrow stripe, streaked brown back, and unstreaked buffy underparts. They measure 5.9 to 6.7 inches in length. The heavy conical bill is adapted for seed cracking. Males vary in the size of their black bib — larger bibs signal dominance.

HABITAT AND RANGE IN CT AND PA
House Sparrows are permanent year-round residents in virtually every town, city, and farm in Connecticut and Pennsylvania. They are among the most abundant birds in both states. They are almost entirely dependent on human structures for nesting and human food sources for survival. You will rarely find them far from buildings. They were introduced in New York in 1851 and have since spread across the continent.

BEHAVIOR
House Sparrows are extremely social and nest colonially. They are aggressive competitors for nest cavities, destroying the eggs and nestlings of native cavity-nesting species including bluebirds, Tree Swallows, and chickadees. Males sing a simple monotonous chirping song from perches on buildings and wires. They dust-bathe enthusiastically in dry soil. Flocks roost communally in dense shrubs and building crevices.

DIET
House Sparrows eat primarily seeds and grains — cracked corn, millet, oats, and bread crumbs. They also eat insects in summer, which they feed to their nestlings. They are highly opportunistic and exploit human food sources including outdoor dining areas and fast-food parking lots.

HOW TO ATTRACT OR DETER
Like Starlings, House Sparrows are considered invasive and most native bird advocates actively try to limit their presence. Using nyjer (thistle) feeders, safflower seeds, and suet-only options reduces House Sparrow use since they prefer millet and cracked corn. Monitoring nest boxes and removing House Sparrow nests protects native cavity nesters. Tube feeders with small ports that exclude their heavy bills also help.

SIMILAR SPECIES
Female House Sparrows can be confused with female Purple Finches and other brown sparrows. The plain unstreaked underparts, buffy eyebrow, and association with buildings help identify female House Sparrows. Native sparrows like Song Sparrows have more distinct facial patterns and breast streaking.
""",

"red_bellied_woodpecker": """
Red-bellied Woodpecker (Melanerpes carolinus)

IDENTIFICATION
Despite its name, the Red-bellied Woodpecker's red belly is barely visible in the field — a faint reddish wash on the lower abdomen. The most visible features are the bold black-and-white barred back (giving it the nickname "ladder-backed" in some regions), the red cap, and the pale face and underparts. In males the red cap extends from the bill all the way to the nape. In females the red is limited to the nape, with a gray crown. They measure 9 to 10.5 inches in length. The long, sticky, barbed tongue can extend up to 2 inches past the bill tip to extract insects and cached food from crevices.

HABITAT AND RANGE IN CT AND PA
Red-bellied Woodpeckers are year-round residents throughout Pennsylvania and increasingly in Connecticut, where their range has been expanding northward over recent decades. They prefer mature deciduous forests, forest edges, orchards, and wooded suburban neighborhoods. They are a common feeder visitor and have become significantly more numerous in New England over the past 30 years.

BEHAVIOR
Red-bellied Woodpeckers are vocal and conspicuous. Their call is a loud, rolling "churr" that carries through the woods and is often the first indication of their presence. Like other woodpeckers they excavate nest cavities in dead wood. They cache food extensively — stuffing seeds, berries, and insects into bark crevices, gaps in wood, and even in the tufts of palm trees. They are bold at feeders and can dominate smaller birds.

DIET
Red-bellied Woodpeckers eat insects, seeds, nuts, berries, and occasionally small lizards and nestling birds. They forage on tree trunks, branches, and at feeders. Acorns and other nuts are cached in large quantities for winter. At feeders they are particularly attracted to suet, sunflower seeds, and whole peanuts.

HOW TO ATTRACT TO YOUR YARD
Suet is the best feeder food for attracting Red-bellied Woodpeckers. They also readily eat sunflower seeds, peanuts, and peanut butter. Providing dead trees or large dead branches (snags) on your property gives them foraging and potential nest sites. They will occasionally use nest boxes but prefer to excavate their own cavities.

SIMILAR SPECIES
The Red-headed Woodpecker has an entirely red head (not just the cap) and large white wing patches — a dramatically different pattern. The Gila Woodpecker and Golden-fronted Woodpecker are southwestern species that do not occur in the East. The barred back and red cap combination makes the Red-bellied distinctive among eastern woodpeckers.
""",

"cedar_waxwing": """
Cedar Waxwing (Bombycilla cedrorum)

IDENTIFICATION
Cedar Waxwings are sleek, elegant birds with soft silky plumage in shades of warm brown, gray, and yellow. Key features: a prominent pointed brown crest, a black mask edged in white, yellow terminal band on the tail, red waxy tips on the secondary wing feathers (which give the species its name), and a bright yellow belly fading to white. They measure 5.5 to 6.7 inches. Their plumage has an almost painted quality — no harsh edges, all blended soft tones. They are one of the few entirely crested birds in eastern North America. Males and females look nearly identical.

HABITAT AND RANGE IN CT AND PA
Cedar Waxwings are year-round residents in Connecticut and Pennsylvania but are highly nomadic, moving constantly in search of fruiting trees and shrubs. Their distribution in any given area depends almost entirely on fruit availability. They breed in open woodlands and forest edges and winter wherever they find abundant berries. Numbers in any particular location vary enormously from year to year.

BEHAVIOR
Cedar Waxwings are almost always found in flocks, sometimes numbering in the hundreds. They have a distinctive high-pitched, thin trilled call — "sreee" — that the whole flock produces as they move. Flocks descend on fruiting trees and strip them almost completely before moving on. They are known to become intoxicated from eating overripe or fermenting berries — intoxicated birds have been found unable to fly and have caused vehicle collisions. They pass berries to each other along a perched row in a courtship ritual.

DIET
Cedar Waxwings are specialist fruit eaters, consuming berries and small fruits almost exclusively. Their favorites include cedar berries (eastern red cedar), crabapples, holly berries, serviceberries, dogwood berries, and hawthorn fruits. In summer they supplement their diet with insects, particularly during breeding season when protein is needed for chick development. They are one of very few birds that can survive on fruit alone.

HOW TO ATTRACT TO YOUR YARD
The best way to attract Cedar Waxwings is to plant fruiting trees and shrubs. Priority plantings for CT and PA include: Eastern red cedar (their namesake food), native crabapples, hawthorns, serviceberry (Amelanchier), native hollies, dogwoods, and viburnums. They do not typically visit seed feeders. A birdbath or moving water source is also attractive. When waxwings find your yard, they may appear suddenly in a large flock and depart just as suddenly.

SIMILAR SPECIES
The Bohemian Waxwing is very similar but larger, with rusty-brown undertail coverts (white in Cedar), yellow and white wing markings, and a gray belly. It is a rare winter visitor to the Northeast during irruption years when food fails to the north. The crested silhouette and social flocking behavior make Cedar Waxwings distinctive when seen well.
""",

"eastern_towhee": """
Eastern Towhee (Pipilo erythrophthalmus)

IDENTIFICATION
Eastern Towhees are large, striking sparrows. Males have a jet black head, back, and breast, bright white belly, and rich rufous-orange sides that extend from the breast to the undertail. White spots on the wings and white outer tail corners are visible in flight. Females replace the black with warm brown but share the same pattern of white belly and rufous sides. The eye is red (some populations have white eyes). They measure 6.8 to 8.2 inches — noticeably larger than most other sparrows. Their size, bold coloring, and loud scratching behavior make them relatively easy to identify.

HABITAT AND RANGE IN CT AND PA
Eastern Towhees are year-round residents in Pennsylvania and summer residents in Connecticut, with some wintering in southern CT. They strongly prefer dense brushy habitats — scrubby second-growth, forest edges with thick underbrush, overgrown fields, and woodland clearings with dense low vegetation. They avoid open lawns and mature closed-canopy forest.

BEHAVIOR
Eastern Towhees are ground foragers with a distinctive two-footed backward scratching technique — they jump forward and scratch back with both feet simultaneously to expose seeds and insects in leaf litter. This produces a loud rustling sound that is often the first indication of their presence in dense brush. The song is a distinctive "drink-your-teeeea" — a short introductory note followed by a long trill. The call is an upslurred "towhee" or "chewink" — both names used for this species historically.

DIET
Eastern Towhees eat seeds, berries, and insects. Seeds and berries dominate in fall and winter; insects and their larvae become more important in summer. They eat acorns, weed seeds, wild berries, beetles, ants, and caterpillars. At feeders they eat white millet and sunflower seeds.

HOW TO ATTRACT TO YOUR YARD
Eastern Towhees are most attracted by habitat — if you have brushy areas with dense low cover, they may already be present. White proso millet scattered on the ground below feeders is the best supplemental food. Brush piles, overgrown hedgerows, and leaving fallen leaves in place (which harbors the invertebrates they scratch for) all support towhees. They are ground feeders and rarely visit elevated feeders.

SIMILAR SPECIES
The Spotted Towhee of the West is very similar with white spots on the black back; it is a rare vagrant in the East. The male Eastern Towhee's bold black-rufous-white pattern is unmistakable among eastern birds. Female Towhees could be confused with other large sparrows but the rufous sides and overall size are distinctive.
""",
"common_grackle" : """
Common Grackle (Quiscalus quiscula)

IDENTIFICATION
Common Grackles are large, long-tailed blackbirds with iridescent plumage that shifts between black, purple, green, and bronze depending on the light angle. Males have a glossy purplish-blue head and breast, a bronzy-green iridescent back and belly, and a distinctive long keel-shaped tail that is folded into a V-shape in flight. Females are similar but smaller, shorter-tailed, and less iridescent. Both sexes have bright golden-yellow eyes — the most striking close-up feature — and a long, slightly downcurved heavy bill. They measure 11 to 13.4 inches in length, making them noticeably larger than a Robin. In flight the long tail and flat-headed silhouette are distinctive.

HABITAT AND RANGE IN CT AND PA
Common Grackles are year-round residents in Pennsylvania and summer residents in Connecticut, arriving in March and departing by November though some linger into winter in southern areas. They are found in open and semi-open habitats including agricultural fields, forest edges, suburban neighborhoods, parks, marshes, and areas near water. They are highly adaptable and thrive in human-modified landscapes. In fall and winter they form enormous mixed blackbird flocks with Red-winged Blackbirds, European Starlings, and Brown-headed Cowbirds that can number in the millions across the mid-Atlantic region.

BEHAVIOR
Common Grackles are bold, assertive, and highly intelligent. They walk on the ground with a purposeful strut, head held high. Males display by puffing up their feathers, spreading their tail into its distinctive keel shape, and producing a loud, harsh, rusty-hinge call — "readle-eak" — that sounds like a creaking gate. They are known to follow farm equipment to catch disturbed insects and to wade into shallow water to catch small fish and crayfish. They are aggressive at feeders and can dominate and exclude smaller birds. Grackles practice a behavior called anting — rubbing ants into their feathers — which is thought to use the formic acid from the ants as a natural insecticide against parasites.

DIET
Common Grackles are highly omnivorous. They eat corn, seeds, insects, earthworms, small fish, crayfish, small frogs, eggs and nestlings of other birds, berries, and human food scraps. Corn is a staple, and they are considered an agricultural pest in some areas due to their consumption of sprouting corn. In suburban areas they are opportunistic scavengers. At feeders they eat sunflower seeds, cracked corn, and millet, often dominating feeders and eating voraciously.

HOW TO ATTRACT OR DETER
Common Grackles need no encouragement — they will find your feeders on their own. Many backyard birders actively try to limit grackle access since they can empty feeders rapidly and intimidate smaller birds. Effective deterrents include: using tube feeders with small ports that exclude their large bills, switching to safflower seeds which grackles find less palatable, using weight-sensitive feeders that close under the weight of large birds, and temporarily removing feeders during peak grackle season in early spring when large flocks pass through. If you enjoy watching grackles, cracked corn on a ground platform is the most effective attractant.

SIMILAR SPECIES
The Boat-tailed Grackle is found along the coast and is larger with an even more dramatically keeled tail. The Great-tailed Grackle is a southwestern species not expected in the Northeast. Brown-headed Cowbirds are smaller and stockier with a shorter tail and brown head on males. European Starlings are smaller with a shorter tail and spotted winter plumage. The Common Grackle's combination of large size, long keeled tail, yellow eye, and iridescent plumage makes it distinctive among eastern blackbirds. In poor light or at a distance, look for the tail shape and eye color to confirm identification.
"""

}

# ══════════════════════════════════════════════════════════════
# WRITE THE DOCUMENTS TO DISK
#
# Loop through every bird in the dictionary and save it
# as its own .txt file in the data/birds/ folder.
# ══════════════════════════════════════════════════════════════

def build_knowledge_base():
    # Keep track of how many files we write
    count = 0

    # Loop through each bird — 'name' is the filename key, 'content' is the text
    for name, content in BIRDS.items():

        # Build the full file path: e.g. "data/birds/american_robin.txt"
        filepath = os.path.join(OUTPUT_DIR, f"{name}.txt")

        # Open the file in write mode ("w") and save the content
        # encoding="utf-8" ensures special characters are handled correctly
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content.strip())  # .strip() removes leading/trailing blank lines

        print(f"  Written: {filepath}")
        count += 1

    print(f"\nDone. {count} bird documents saved to {OUTPUT_DIR}/")

# ── Entry point ────────────────────────────────────────────────
# This block only runs when you execute this file directly.
# If another script imports this file, this block is skipped.
if __name__ == "__main__":
    print("Building birding knowledge base...\n")
    build_knowledge_base()