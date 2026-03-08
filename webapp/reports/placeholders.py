# webapp/report/placeholders.py

from reportlab.platypus import Paragraph, Spacer, PageBreak


def build_under_development_page(story, styles, title: str) -> None:
    story.append(Paragraph(title, styles["H1"]))
    story.append(Spacer(1, 60))
    story.append(
        Paragraph(
            "This section is currently under development and will be included in an upcoming version of the Coach Assistant Report.",
            styles["Small"],
        )
    )
    story.append(PageBreak())