from preprocessing import checkin
from config import gowalla, brightkite, weeplaces


def pipeline(c):
    #preprocessing
    check_in_preprocess = checkin.preprocess_totalcheckin(c.origin_check_in, c.check_in_remove_sparseplace, c)
    check_in_preprocess = checkin.checkin_preprocess(c.check_in_remove_sparseplace, c.check_in_preprocess)


if __name__=='__main__':
    pipeline(gowalla)
    pipeline(brightkite)
    pipeline(weeplaces)
