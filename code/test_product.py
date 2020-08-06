import unittest

from image_recognition.serverless.trainers.text.product import NGramToProduct
from image_recognition.serverless.trainers.text.product import NCharToProduct


class TestTrainerTextToProduct(unittest.TestCase):

    def test_trainerProduct(self):
        pruduct_trainer = NGramToProduct(limit=10)
        model = pruduct_trainer.train()
        pruduct_trainer.force_save()
        del pruduct_trainer

        pruduct_trainer = NGramToProduct().force_load()
        new_model = pruduct_trainer
        for c1, c2 in zip(model.model.coef_, new_model.model.coef_):
            self.assertTrue(all(c1 == c2))

    def test_functionBrand(self):

        x_raw = """LaserMax Red Laser Internal Guide Rod Laser Sight For Glock 19, 23, : LMS-1131P
LaserMax Red Laser Internal Guide Rod Laser Sight For Glock 19, 23, : LMS-1131P
LaserMax Red Laser Internal Guide Rod Laser Sight For Glock 19, 23, : LMS-1131P
Teva Langdon Sandal (Mens). 1015149.
Teva Langdon Sandal (Mens). 1015149.
Teva Langdon Sandal (Mens). 1015149.
BSN Syntha-6 Ultra Premium Sustained Release Protein 2.91 lbs Cookies & Cream
BSN Syntha-6 Ultra Premium Sustained Release Protein 2.91 lbs Cookies & Cream
Enbrighten Cafe LED String Lights, Black, 36-Foot, 18 Bulbs, Weatherproof, Shatterproof, Commercial Grade, 33171
Enbrighten Cafe LED String Lights, Black, 36-Foot, 18 Bulbs, Weatherproof, Shatterproof, Commercial Grade, 33171
Enbrighten Cafe LED String Lights, Black, 36-Foot, 18 Bulbs, Weatherproof, Shatterproof, Commercial Grade, 33171
Enbrighten Cafe LED String Lights, Black, 36-Foot, 18 Bulbs, Weatherproof, Shatterproof, Commercial Grade, 33171
Enbrighten Cafe LED String Lights, Black, 36-Foot, 18 Bulbs, Weatherproof, Shatterproof, Commercial Grade, 33171
Enbrighten Cafe LED String Lights, Black, 36-Foot, 18 Bulbs, Weatherproof, Shatterproof, Commercial Grade, 33171
Enbrighten Cafe LED String Lights, Black, 36-Foot, 18 Bulbs, Weatherproof, Shatterproof, Commercial Grade, 33171
Enbrighten Cafe LED String Lights, Black, 36-Foot, 18 Bulbs, Weatherproof, Shatterproof, Commercial Grade, 33171
K N Engineering K N Air Filter Honda VFR800 Interceptor,VF
K N Engineering K N Air Filter Honda VFR800 Interceptor,VF
K N Engineering K N Air Filter Honda VFR800 Interceptor,VF
K N Engineering K N Air Filter Honda VFR800 Interceptor,VF
Beretta Beretta Apx Compact 9Mm 3.7 13Rd JAXC921"
Beretta Beretta Apx Compact 9Mm 3.7 13Rd JAXC921"
Beretta Beretta Apx Compact 9Mm 3.7 13Rd JAXC921"
Beretta Beretta Apx Compact 9Mm 3.7 13Rd JAXC921"
Edelbrock Engine Camshaft & Lifter Kit 7122; Hydraulic for Ford 289/302
Edelbrock Engine Camshaft & Lifter Kit 7122; Hydraulic for Ford 289/302
Edelbrock Engine Camshaft & Lifter Kit 7122; Hydraulic for Ford 289/302
Edelbrock Engine Camshaft & Lifter Kit 7122; Hydraulic for Ford 289/302
Edelbrock Engine Camshaft & Lifter Kit 7122; Hydraulic for Ford 289/302
Panasonic Lumix DMC-LX100 Digital Camera, Silver #DMC-LX100S
Panasonic Lumix DMC-LX100 Digital Camera, Silver #DMC-LX100S
Panasonic Lumix DMC-LX100 Digital Camera, Silver #DMC-LX100S
Panasonic Lumix DMC-LX100 Digital Camera, Silver #DMC-LX100S
Panasonic Lumix DMC-LX100 Digital Camera, Silver #DMC-LX100S
Panasonic Lumix DMC-LX100 Digital Camera, Silver #DMC-LX100S""".split("\n")

        y_raw = """30202
30202
30202
47479
47479
47479
49652
49652
51494
51494
51494
51494
51494
51494
51494
51494
69827
69827
69827
69827
106190
106190
106190
106190
110546
110546
110546
110546
110546
118042
118042
118042
118042
118042
118042""".split("\n")

        y_raw = list(map(int, y_raw))

        brand_trainer = NGramToProduct()
        model = brand_trainer.train_x_y(x_raw, y_raw)

        for x, y_r in zip(x_raw, y_raw):
            y_p = model.predict(x).pop()
            self.assertEquals(y_p, y_r)


class TestTrainerNCharTextToProduct(unittest.TestCase):

    def test_trainer_NCharProduct(self):
        product_trainer = NCharToProduct(limit=10)
        model = product_trainer.train()
        product_trainer.force_save()
        del product_trainer

        product_trainer = NCharToProduct().force_load()
        new_model = product_trainer
        for c1, c2 in zip(model.model.coef_, new_model.model.coef_):
            self.assertTrue(all(c1 == c2))

    def test_function_NCharBrand(self):

        x_raw = """Renew Life Critical Colon 80 Billion 30 Vegetable Caps
Renew Life Colon Care Ultimate Flora Probiotic 80 Billion 60 Vegetable Capsules
Ultimate Flora Critical Colon Bifidomax 80 Billion
Renew Life Critical Colon 80 Billion 60 Vegetable Caps
Renew Life Ultimate Flora RTS Women's Probiotic 15 Billion 30 Veggie Capsules
UF WOMEN'S CARE GO PACK 15B 30CT
Ultimate Flora Rts Colon Care Probiotic 15 Billion
Ultimate Flora Rts Daily Probiotic 15 Billion
Ultimate Flora Constipation Relief 30 Billion
Ultimate Flora Kids Probiotic
Baby's Jarro Dophilus 2.5oz
Baby's Jarro Dophilus
Baby's Jarro-Dophilus+FOS Powder 2.5 oz by Jarrow Formulas (F) One Time Delivery —
Jarrow Formulas Whey Protein
Whey Protein Vanilla 2lbs
Whey Protein
Hyaluronic Acid
Hyaluronic Acid 60 Caps
Alvita Teas Organic Herbal Dandelion Tea - 24 Tea Bags
Teabags Dandelion Root Organic 24 ct
Organic Dandelion Root Tea
Alvita Teas Echinacea And Goldenseal Tea - Organic - 24 Tea Bags
Echinacea Goldenseal Tea
Alvita Teas Organic Echinacea & Goldenseal Tea Caffeine Free 24 Tea Bags 1.69 oz (48 g)
Echinacea Tea Organic Alvita Tea 24 Bag
Elder Flower Tea Organic 24 Bags
Taurine 1000mg Mega
Mega Taurine Caps
Mega Taurine
Mega Taurine Caps by Twinlab
Mega Taurine 50 Caps Twinlab
Twinlab Mega Taurine CAPS 1000 mg 50 Capsules
TWL Melatonin (3 Mg) Caps 60
Melatonin
Twinlab Melatonin Caps - 3 mg - 60 Capsules
TwinLab Melatonin Caps 60 capsules
Melatonin by Twinlab
Sig Sauer 238380BSS P238 380 ACP 2.7 6+1 NS Poly Grip Black"
Sig Sauer P238 Aluminum Grips NS
Sig Sauer P238 Nitron Carry .380 w/ 2.7 Barrel"
SIG SAUER 238-380-BSS 380acp
Sig Sauer P238 .380acp 2.7 Blk/Blk"
SIG SAUER 238 P238 NITRON NIGHT SIGHTS 238-380-BSS
Sig Sauer P238 Nitron 380 6rd
Sig Sauer P238 238 Pink Pearl 380 Auto Nights 6rd
Sig Sauer P238 Pink Pearl 380 ACP 6+1 Pink Pearlite Grips Nitron
P238
Sig Sauer 1911 Nitron .45 ACP #1911-45-BSS
Sig Sauer P238 238 ESP
Sig P238 380acp 2.7 6rd Pink"
Sig Sauer P238 Engraved with Pink Pearl Grips .380 ACP #238-380-BSS-ESP
Sig Sauer P238 Pink Pearl 380 238-380-BSS-ESP
Sig Sauer P238
Sig Sauer P238 Semi-automatic Single Action Only Compact 380ACP 2.7 Alloy Tan Hogue 6Rd 1 Mag Fixed Night Sights 238-380-DES"
Sig Sauer P238 238 238-380-DES .380 ACP
Sig Sauer 238380desamb P238 Desert 380
P238
Sig Sauer P238 238 DES Tan
Sig Sauer P238 238 Desert 380 Auto 7rd Mag Nights
Sig Sauer P238 Desert 380 ACP 2.7 7+1Rds"
SIG SAUER 238 P238 DESERT .380 ACP 238-380-DES
Sig SAuer P238 Desert .380 ACP Pistol 238-380-DES""".split("\n")

        y_raw = """631257158741
631257158741
631257158741
631257158741
631257158727
631257158727
631257158710
631257158703
631257158697
631257158680
790011030133
790011030133
790011030133
790011210030
790011210030
790011210030
790011290186
790011290186
27434038164
27434038164
27434038164
27434039291
27434039291
27434039291
27434039277
27434037617
27434001984
27434001984
27434001984
27434001984
27434001984
27434001984
27434005111
27434005111
27434005111
27434005111
27434005111
798681415212
798681415212
798681415212
798681415212
798681415212
798681415212
798681415212
798681439317
798681439317
798681439317
798681439317
798681439317
798681439317
798681439317
798681439317
798681439317
798681438020
798681438020
798681438020
798681438020
798681438020
798681438020
798681438020
798681438020
798681438020""".split("\n")

        y_raw = list(map(int, y_raw))

        product_trainer = NCharToProduct()
        model = product_trainer.train_x_y(x_raw, y_raw)

        #for x, y_r in zip(x_raw, y_raw):
        #    y_p = model.predict(x).pop()
        #    self.assertEquals(y_p, y_r)

        y_predicted = model.predict(x_raw)

        self.assertEquals(set(y_predicted), set(y_raw))


if __name__ == '__main__':
    unittest.main()
