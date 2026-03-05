import schemdraw
from schemdraw.flow import *


def generate_auth_flow():
    print("Generating Auth Flowchart...")
    # FIX: Replaced '&' with 'and' to prevent XML ParseError
    with schemdraw.Drawing(file='flowchart_auth.png', show=False) as d:
        d.config(fontsize=11)

        d.add(Start(w=3).label("Start"))
        d.add(Arrow().down(d.unit / 2))

        # Changed '&' to 'and' here
        d.add(Box(w=4).label("Input User and Pass\n+ Stored Hash"))
        d.add(Arrow().down(d.unit / 2))

        d.add(Box(w=4).label("Hash Input Password\n(SHA-256)"))
        d.add(Arrow().down(d.unit / 2))

        decision = d.add(Decision(w=5).label("Hash == Stored?"))

        # Yes Path
        d.add(Arrow().right(d.unit).at(decision.E).label("Yes"))
        true_box = d.add(Box(w=3).label("Return Hash\n(Login Success)"))
        d.add(Arrow().down(d.unit).at(true_box.S))

        end = d.add(Start(w=3).label("End"))

        # No Path
        d.add(Arrow().down(d.unit).at(decision.S).label("No"))
        false_box = d.add(Box(w=3).label("Return False\n(Login Fail)"))
        d.add(Line().right(d.unit * 1.3).at(false_box.E))  # Connect visual line to end
        d.add(Arrow().to(end.E))


def generate_sentiment_flow():
    print("Generating Sentiment Flowchart...")
    with schemdraw.Drawing(file='flowchart_sentiment.png', show=False) as d:
        d.config(fontsize=11)

        d.add(Start(w=3).label("Input Score"))
        d.add(Arrow().down(d.unit / 2))

        dec1 = d.add(Decision(w=4).label("Score > 0.05?"))

        # Positive
        d.add(Arrow().right(d.unit * 1.5).at(dec1.E).label("Yes"))
        pos = d.add(Box(w=3).label("Return 'Positive'"))

        # No -> Next Decision
        d.add(Arrow().down(d.unit).at(dec1.S).label("No"))
        dec2 = d.add(Decision(w=4).label("Score < -0.05?"))

        # Negative
        d.add(Arrow().right(d.unit * 1.5).at(dec2.E).label("Yes"))
        neg = d.add(Box(w=3).label("Return 'Negative'"))

        # Neutral
        d.add(Arrow().down(d.unit).at(dec2.S).label("No"))
        neu = d.add(Box(w=3).label("Return 'Neutral'"))

        # Connect to End
        d.add(Arrow().down(d.unit / 2).at(neu.S))
        end = d.add(Start(w=3).label("End"))

        # Draw lines connecting Side branches to End
        d.add(Line().down(d.unit * 3.5).at(pos.S))  # Long line down
        d.add(Arrow().to(end.E))

        d.add(Line().down(d.unit * 1.2).at(neg.S))
        d.add(Arrow().to(end.E))


def generate_scraper_flow():
    print("Generating Scraper Flowchart...")
    with schemdraw.Drawing(file='flowchart_scraper.png', show=False) as d:
        d.config(fontsize=11)

        d.add(Start(w=3).label("Start"))
        d.add(Arrow().down(d.unit / 2))

        # FIX: Ensure no special chars here either
        req = d.add(Box(w=4).label("Request URL\nand Parse HTML"))
        d.add(Arrow().down(d.unit / 2))

        loop = d.add(Decision(w=4).label("Item Found?"))

        # Loop Body
        d.add(Arrow().right(d.unit).at(loop.E).label("Yes"))
        extract = d.add(Box(w=3).label("Extract Data"))
        d.add(Arrow().down(d.unit).at(extract.S))
        append = d.add(Box(w=3).label("Append List"))

        # Loop Back
        d.add(Line().left(d.unit / 2).at(append.W))
        d.add(Line().up(d.unit * 2.5))
        d.add(Arrow().to(loop.N))

        # Loop Exit
        d.add(Arrow().down(d.unit).at(loop.S).label("No"))
        d.add(Box(w=4).label("Create DataFrame"))
        d.add(Arrow().down(d.unit / 2))
        d.add(Start(w=3).label("End"))


if __name__ == "__main__":
    generate_auth_flow()
    generate_sentiment_flow()
    generate_scraper_flow()
    print("Success: Generated flowchart_auth.png, flowchart_sentiment.png, flowchart_scraper.png")