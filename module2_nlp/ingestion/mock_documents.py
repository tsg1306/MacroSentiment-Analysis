from datetime import datetime


def get_mock_documents() -> list:
    return [
        {
            "title": "Bridgewater Q3 2024 Client Letter",
            "source": "Bridgewater Q3 Letter",
            "doc_type": "hedge_fund_letter",
            "published_at": datetime(2024, 10, 15),
            "text": (
                "Dear Investors, as we close Q3 2024 we want to share our updated macro outlook "
                "and portfolio positioning. The global economy continues to navigate a complex "
                "landscape of persistent inflation, geopolitical tensions, and shifting monetary "
                "policy regimes across major central banks. "
                "Our conviction in gold as a strategic allocation has strengthened considerably. "
                "Central bank gold purchases have reached multi-decade highs, with China, India, "
                "and several emerging market central banks accelerating their diversification away "
                "from dollar reserves. The structural bid for gold from official sector buyers "
                "represents a paradigm shift that we believe will support prices well above "
                "current levels over the medium term. We have increased our gold allocation by "
                "fifteen percent this quarter. "
                "Conversely, we have reduced our equities exposure significantly. The S&P 500 and "
                "NASDAQ are trading at valuations that imply historically optimistic earnings growth "
                "trajectories. With the Federal Reserve maintaining a hawkish stance and credit "
                "conditions tightening, we see downside risk to corporate margins. The equity risk "
                "premium has compressed to levels last seen before the 2008 crisis. Stocks appear "
                "vulnerable to a correction of ten to fifteen percent. "
                "In fixed income, we have shifted duration shorter as we expect the yield curve to "
                "remain inverted through Q1 2025. Real rates are likely to stay elevated, which "
                "historically correlates with underperformance in growth equities and outperformance "
                "in real assets like gold and commodities. "
                "We also note rising geopolitical risk premia across energy markets. The conflict "
                "dynamics in the Middle East and sanctions on Russian oil continue to create supply "
                "uncertainty. While we are not directly positioned for an oil spike, we acknowledge "
                "the tail risk. Our portfolio remains defensively positioned with an overweight in "
                "gold, underweight in equities, and selective commodity exposure."
            ),
        },
        {
            "title": "Iran Nuclear Deal Revival Sends Oil Prices Tumbling",
            "source": "Reuters",
            "doc_type": "news",
            "published_at": datetime(2024, 9, 12),
            "text": (
                "LONDON — Oil prices fell sharply on Thursday after Iran and Western powers "
                "announced a breakthrough in nuclear negotiations, raising expectations that "
                "Iranian crude exports could return to global markets within months. "
                "Brent crude dropped four percent to seventy-one dollars per barrel, while WTI "
                "fell to sixty-seven dollars, the lowest level since March. The sell-off "
                "accelerated in afternoon trading as traders priced in the potential return of "
                "up to one point five million barrels per day of Iranian supply. "
                "The agreement, reached after months of backchannel diplomacy, would see Iran "
                "limit its uranium enrichment program in exchange for phased sanctions relief. "
                "Under the proposed timeline, oil export restrictions would begin easing in Q1 "
                "2025, with full normalization expected by mid-year. "
                "OPEC has signaled it may need to revisit its production quotas if Iranian "
                "barrels return to the market. Saudi Arabia, which has been voluntarily cutting "
                "production to support prices, faces a dilemma between maintaining market share "
                "and defending its fiscal breakeven price of approximately eighty dollars per barrel. "
                "Energy analysts at Goldman Sachs revised their Brent forecast down by eight "
                "dollars to seventy-five dollars for Q4 2024, citing the Iran supply overhang. "
                "Meanwhile, oil inventory data showed a surprise build of three million barrels, "
                "adding to bearish sentiment. The crude oil market appears oversupplied heading "
                "into the winter season, which could further weigh on prices."
            ),
        },
        {
            "title": "Federal Reserve FOMC Minutes November 2024",
            "source": "Federal Reserve",
            "doc_type": "fomc_minutes",
            "published_at": datetime(2024, 11, 7),
            "text": (
                "Minutes of the Federal Open Market Committee meeting held on November 6-7, 2024. "
                "Participants noted that inflation remained above the Committee's two percent "
                "target despite twelve months of restrictive monetary policy. Core PCE inflation "
                "was running at three point one percent, driven by persistent services inflation "
                "and a resilient labor market. "
                "Several participants expressed concern that premature easing could reignite "
                "inflationary pressures, particularly given elevated wage growth in the services "
                "sector. The Committee voted unanimously to maintain the federal funds rate at "
                "five point two five to five point five percent, with forward guidance indicating "
                "rates would remain higher for longer than previously anticipated. "
                "The staff presented updated economic projections showing GDP growth slowing to "
                "one point four percent in 2025, below trend. The unemployment rate was projected "
                "to rise to four point eight percent by mid-2025 as the lagged effects of "
                "monetary tightening continue to work through the economy. "
                "Financial conditions have tightened notably since September, with the S&P 500 "
                "declining eight percent and high-yield credit spreads widening by sixty basis "
                "points. Several participants noted that equity market valuations, while lower, "
                "still did not fully reflect the deterioration in the earnings outlook. The SPX "
                "forward price-to-earnings ratio remained above historical averages. "
                "The Committee discussed risks to the outlook, including geopolitical tensions, "
                "a potential government shutdown, and the impact of higher interest rates on "
                "commercial real estate. Participants agreed that the balance of risks had "
                "shifted toward overtightening, but concluded that maintaining the current "
                "stance was appropriate given persistent inflation readings."
            ),
        },
        {
            "title": "ExxonMobil Q3 2024 Earnings Call Transcript",
            "source": "ExxonMobil Q3 2024",
            "doc_type": "earnings_call",
            "published_at": datetime(2024, 10, 25),
            "text": (
                "Good morning and welcome to ExxonMobil's third quarter 2024 earnings conference "
                "call. I'm Darren Woods, Chairman and CEO. Today we reported earnings of nine "
                "point two billion dollars, roughly in line with analyst expectations but down "
                "twelve percent from the prior quarter. "
                "Starting with upstream, production averaged three point seven million barrels "
                "of oil equivalent per day, a new record driven by continued ramp-up in the "
                "Permian Basin and Guyana. However, realized crude oil prices were lower "
                "sequentially, with WTI averaging seventy-three dollars compared to eighty-one "
                "dollars in Q2. We see oil markets as broadly balanced heading into 2025, with "
                "OPEC discipline offsetting modest demand growth. "
                "On the demand side, we are closely watching developments in China. Chinese "
                "crude imports declined for the second consecutive quarter as economic activity "
                "slows. The property sector downturn continues to weigh on industrial demand "
                "for petrochemicals and diesel. China's GDP growth has slowed to four point "
                "two percent, below government targets, and we do not expect a meaningful "
                "recovery before the second half of 2025. "
                "In our downstream segment, refining margins compressed significantly as gasoline "
                "demand softened and inventory builds pressured crack spreads. We expect this "
                "trend to persist through Q4. Our chemical segment saw modest improvement in "
                "polyethylene margins but overall remains challenged by global overcapacity. "
                "Capital expenditure for the full year is expected to be approximately twenty-five "
                "billion dollars, consistent with our guidance. We remain committed to our "
                "Permian Basin and Guyana growth programs."
            ),
        },
        {
            "title": "JPMorgan Gold Outlook: Upgrading to Overweight",
            "source": "JPMorgan Research",
            "doc_type": "research_note",
            "published_at": datetime(2024, 8, 20),
            "text": (
                "We are upgrading gold to Overweight from Neutral in our global commodity "
                "allocation, with a twelve-month price target of twenty-four hundred dollars "
                "per ounce, representing fifteen percent upside from current levels. "
                "Three structural drivers underpin our bullish thesis on gold. First, central "
                "bank purchases have fundamentally altered the supply-demand equation. Official "
                "sector buying has averaged over one thousand tonnes annually for the past two "
                "years, led by China's PBOC and the Reserve Bank of India. We expect this trend "
                "to accelerate as de-dollarization efforts gain momentum. "
                "Second, the approaching Federal Reserve rate cutting cycle provides a powerful "
                "tailwind. Historically, gold has rallied an average of twenty percent in the "
                "twelve months following the first rate cut. With real rates likely peaking in "
                "Q4 2024, the opportunity cost of holding gold is set to decline meaningfully. "
                "Third, geopolitical risk premia remain elevated. The ongoing conflicts in "
                "Ukraine and the Middle East, combined with US-China tensions over Taiwan, "
                "support safe haven demand. Our geopolitical risk index is at its highest level "
                "since 2020, which historically correlates with gold outperformance. "
                "We note potential headwinds including a stronger US dollar scenario and the "
                "possibility of a hawkish surprise from the Fed. However, we believe the "
                "risk-reward is compelling at current levels. Gold mining equities, particularly "
                "Newmont and Barrick, offer leveraged exposure to our bullish gold view. "
                "Investors should consider a five to ten percent strategic allocation to gold "
                "as a portfolio diversifier and inflation hedge."
            ),
        },
        {
            "title": "OPEC+ Announces Surprise Production Cut of 1.5M bpd",
            "source": "Bloomberg",
            "doc_type": "news",
            "published_at": datetime(2024, 11, 30),
            "text": (
                "VIENNA — OPEC+ announced an unexpected additional production cut of one point "
                "five million barrels per day starting January 2025, sending oil prices surging "
                "in after-hours trading. Brent crude jumped six percent to eighty-four dollars "
                "per barrel, while WTI rose to seventy-nine dollars. "
                "The decision, led by Saudi Arabia which will shoulder the largest share of cuts "
                "at seven hundred thousand barrels per day, caught markets off guard. Most "
                "analysts had expected the group to maintain existing quotas at their meeting "
                "in Vienna. Russia agreed to cut an additional two hundred thousand barrels per "
                "day, bringing total OPEC+ cuts to approximately five point five million bpd. "
                "Saudi Energy Minister Prince Abdulaziz bin Salman said the cuts were a "
                "preemptive measure to ensure market stability heading into 2025. He emphasized "
                "that the group remains committed to balancing supply and demand, and warned "
                "speculators against betting on lower prices. "
                "The surprise move is seen as a response to weakening demand signals from China "
                "and concerns about a potential global economic slowdown. Oil prices had been "
                "trending lower since October, with WTI briefly touching sixty-five dollars. "
                "Goldman Sachs immediately revised its Q1 2025 Brent forecast upward by ten "
                "dollars to ninety dollars per barrel, calling the cuts a game-changer for "
                "market balances. Energy stocks rallied in pre-market trading, with the XLE "
                "ETF up three percent. The oil market now faces a significant supply deficit "
                "in early 2025 that could push crude prices toward triple digits if demand "
                "holds up and OPEC+ maintains compliance."
            ),
        },
    ]
