from civiclens.nlp import NaiveBayesIssueClassifier


def test_classifier_recognizes_common_report_categories() -> None:
    classifier = NaiveBayesIssueClassifier()

    assert classifier.predict("A deep pothole is damaging cars").category == "roads"
    assert classifier.predict("Garbage bins are overflowing").category == "waste"
    assert classifier.predict("A live electrical wire is sparking").category == "public_safety"


def test_unrecognized_report_has_low_confidence() -> None:
    prediction = NaiveBayesIssueClassifier().predict(
        "The purple moon sings quietly"
    )

    assert prediction.confidence < 0.38
