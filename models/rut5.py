# https://huggingface.co/cointegrated/rut5-base-multitask
def rut5_prepare(abstract):
    """
    Adds 'headline | ' prefix

    @param abstract: string, abstract to preprocess
    """
    abstract_ = 'headline | ' + abstract
    return abstract_
