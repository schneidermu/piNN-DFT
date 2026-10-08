# Publication dataset v1 build report

Status: qualified local training/validation bundle.

```text
publication_dataset_v1/
  dataset_manifest.json / splits.json / checksums.sha256 / DATASET_CARD.md
  provenance/  schema, source identities, build/qualification receipts, leakage audit
  chemistry/   reactions.jsonl, species.jsonl, chemistry_*.h5
  mrks/        systems.jsonl, mrks_*.h5
  validation/  diet30_reactions.jsonl, diet30_species.jsonl, validation_*.h5
```

## Counts

```json
{
  "chemistry": {
    "ae17": 17,
    "databases": {
      "ABDE4": 4,
      "AE17": 17,
      "DBH76": 70,
      "EA13": 11,
      "IP13": 13,
      "MGAE109": 104,
      "NCCE31": 28,
      "PA8": 8,
      "pTC13": 13
    },
    "identities": 268,
    "relchem": 251,
    "source_groups": [
      {
        "file": "groups/group_234_key_250_ABDE4_0.pickle",
        "sha256": "160cb895e2f38d74d095fe6bb4dbc1180dcfedb0b26cee6a7a2cd395de7e4011"
      },
      {
        "file": "groups/group_235_key_251_ABDE4_1.pickle",
        "sha256": "dd411ec559abc4a38ba56557443020e7504979d2a3109d754480015e8e805ed6"
      },
      {
        "file": "groups/group_236_key_252_ABDE4_2.pickle",
        "sha256": "c526bab4b5df68fdd3c8cfd598722414353b9e4229ba3db32c929eef23407e04"
      },
      {
        "file": "groups/group_237_key_253_ABDE4_3.pickle",
        "sha256": "63102bc6387f192affe95742b6aad8c312e610b1b7b28474877e43121f8ee792"
      },
      {
        "file": "groups/group_238_key_254_AE17_0.pickle",
        "sha256": "d3159496190bae06f9adeeb26711ed19959deb79644a1b2b30906fe21593ccbd"
      },
      {
        "file": "groups/group_239_key_255_AE17_1.pickle",
        "sha256": "210d363ee907269a86181aa7eb5efabc5a7d15331a7599da43c82e1bf8010d25"
      },
      {
        "file": "groups/group_240_key_256_AE17_2.pickle",
        "sha256": "a6178c8e504ba231769d82f2f9fd4a72063e07530fb04c908c95dd7ba38bfc5f"
      },
      {
        "file": "groups/group_241_key_257_AE17_3.pickle",
        "sha256": "cce757a5ed13f2921a3a626a6b3ac3e3242ac48214e827f93c2c454a4f8a666b"
      },
      {
        "file": "groups/group_242_key_258_AE17_4.pickle",
        "sha256": "0a0bc86b6a85fe7748ef114e999d5fddc98008040de5523f4335fb11dce8c7a2"
      },
      {
        "file": "groups/group_243_key_259_AE17_5.pickle",
        "sha256": "83d7ee5bc922988b168bda0a1f864e3f560f8ad955068713c072d8d4c9c5b866"
      },
      {
        "file": "groups/group_244_key_260_AE17_6.pickle",
        "sha256": "dc63e457d955bdb52ffd921365b2eeb95b3b39d71c55274fc3ba4f89553460f4"
      },
      {
        "file": "groups/group_245_key_261_AE17_7.pickle",
        "sha256": "8a3eb654e1468a520b7254981a8441e276447069ed04c9f8a66b8b8fc3e62fa4"
      },
      {
        "file": "groups/group_246_key_262_AE17_8.pickle",
        "sha256": "066e3954d1cf282cd4da3fc96256346fbe822a433ae597d924e2fa34d18133a4"
      },
      {
        "file": "groups/group_247_key_263_AE17_9.pickle",
        "sha256": "61b7701a5413a8cf4e9640784781de01d5fdec196a51224842789b9bc7d4d4ac"
      },
      {
        "file": "groups/group_248_key_264_AE17_10.pickle",
        "sha256": "69d84d785f88a71276069c32e0363a0d291229890040eaacd4c2985f79048701"
      },
      {
        "file": "groups/group_249_key_265_AE17_11.pickle",
        "sha256": "de68be81b2c9fe8cefba101765c4012a77500e54cb3f3f5ac8e4cc185e6313bd"
      },
      {
        "file": "groups/group_250_key_266_AE17_12.pickle",
        "sha256": "b29053561225a6c60eef9f69f60ea7747b5562da7f89bfd90200ec97f6456251"
      },
      {
        "file": "groups/group_251_key_267_AE17_13.pickle",
        "sha256": "49c28aaf19ddfc79fb0e83b4c5ba96a09d00ef02b39372976672f87cf13a3a61"
      },
      {
        "file": "groups/group_252_key_268_AE17_14.pickle",
        "sha256": "e15c55432cf42869972e23a13a284342842dee02e91b091703001aa7657d4d76"
      },
      {
        "file": "groups/group_253_key_269_AE17_15.pickle",
        "sha256": "5761ffa571e3ed5ac05929857d91d1c1edbfaf5c4748675cbc173f1a6400e5c1"
      },
      {
        "file": "groups/group_254_key_270_AE17_16.pickle",
        "sha256": "12821477c6e359c46721321bb227bfcf7c0d31e7f562cbf0b9bd81c5c862f478"
      },
      {
        "file": "groups/group_136_key_143_DBH76_0.pickle",
        "sha256": "3e3d5061fdaa553ef61937aa1dfac92a8150c3e9984c871392461943d2bd2187"
      },
      {
        "file": "groups/group_137_key_144_DBH76_1.pickle",
        "sha256": "7d6ccfe5b49200a509cfbbce7c62bf34cdde89188a360eec7476095c9dcdc03a"
      },
      {
        "file": "groups/group_138_key_145_DBH76_2.pickle",
        "sha256": "cecca2ddcbdff975abe5410118d63e655b4e27a050d14511b1dd29d96fb9abbe"
      },
      {
        "file": "groups/group_139_key_146_DBH76_3.pickle",
        "sha256": "7e8fb9327efc8ecd768be31e13e58ef0b5215f528b27b5f18ef4a2fd44955039"
      },
      {
        "file": "groups/group_140_key_147_DBH76_4.pickle",
        "sha256": "58a58b3462360b14f49c7ed01e0a5f5837e8527e9fe58391551979dd3b7d467b"
      },
      {
        "file": "groups/group_141_key_148_DBH76_5.pickle",
        "sha256": "a7a983c0b3d2381d424b536a090200aef26156afb80012a4729520c34c011b96"
      },
      {
        "file": "groups/group_142_key_149_DBH76_6.pickle",
        "sha256": "fba92202125842d0c4c78184f40181fb90f49b462551d6adc6d02c494b969023"
      },
      {
        "file": "groups/group_143_key_150_DBH76_7.pickle",
        "sha256": "5def6b91a1d4f0d3976e87e65c5fb70f4ab65380a30798fee404c85bf6b0a62f"
      },
      {
        "file": "groups/group_144_key_151_DBH76_8.pickle",
        "sha256": "cb413276d6a7fd04820c2639faea911597800e563d35c82f515917987951a59b"
      },
      {
        "file": "groups/group_145_key_152_DBH76_9.pickle",
        "sha256": "7cf02a82f630d1f0ecbe90b2dfa942b5bfe85022b1f5fe2b68ee1bad5294affc"
      },
      {
        "file": "groups/group_146_key_153_DBH76_10.pickle",
        "sha256": "260efb8032873068b4a30443f4b100b7b476e5b420ed63a80f31626381ee8b39"
      },
      {
        "file": "groups/group_147_key_154_DBH76_11.pickle",
        "sha256": "bd0a3cc8704c7a8447d2a63c168612479c75e59da4d3fabb242317509e9abc8f"
      },
      {
        "file": "groups/group_148_key_155_DBH76_12.pickle",
        "sha256": "340701c838848a687cd2302ac186064948a7448c3a0a120472931de9bd5dded9"
      },
      {
        "file": "groups/group_149_key_156_DBH76_13.pickle",
        "sha256": "06017883bda19556b3efd88f2f5672b623a0914bc756f666226cda1f65c7355e"
      },
      {
        "file": "groups/group_150_key_157_DBH76_14.pickle",
        "sha256": "e2cac319b4be52e0b091eed3c06a20a625030fd57ecd8f519648fecb09c6b4bc"
      },
      {
        "file": "groups/group_151_key_159_DBH76_16.pickle",
        "sha256": "a5c26b8f7c73dd180e822e9109a24f48bebe7e886b768aa09c6d837af2bc3e2f"
      },
      {
        "file": "groups/group_152_key_160_DBH76_17.pickle",
        "sha256": "2e4db68b25b25a597636839a35565a81d95db932872e88406b581755be54ce9c"
      },
      {
        "file": "groups/group_153_key_161_DBH76_18.pickle",
        "sha256": "e15ca4d707ccfe87f83963a2bdd44a0220c93a8453a1a457030cc811acd4d4d7"
      },
      {
        "file": "groups/group_154_key_162_DBH76_19.pickle",
        "sha256": "4493106aa8a668e895265f3dc8f345818be1539f4df9559768bcbb9ff56579b2"
      },
      {
        "file": "groups/group_155_key_163_DBH76_20.pickle",
        "sha256": "03fdc1eafd51c95ff9cabb63633556917c793d73660080d4f1a570e4b5199e3e"
      },
      {
        "file": "groups/group_156_key_164_DBH76_21.pickle",
        "sha256": "0247be1306854802f14438eb148d6af5430ea655583e8e8efe27acb10deb9fe8"
      },
      {
        "file": "groups/group_157_key_165_DBH76_22.pickle",
        "sha256": "6d99546b4830163fef8636cbc10669aa3e73a78a4b797bb36d00bfebd93cd0d8"
      },
      {
        "file": "groups/group_158_key_166_DBH76_23.pickle",
        "sha256": "91190bad897192230fa29029f7b9da7b37674da6e2331a7108ea6fc3ec5c68b1"
      },
      {
        "file": "groups/group_159_key_167_DBH76_24.pickle",
        "sha256": "7be69b21dca2c76a46ce016e68a16f2786d5b68f9b3a5dd48bb2aa96d873324c"
      },
      {
        "file": "groups/group_160_key_168_DBH76_25.pickle",
        "sha256": "7fe38f77cb501f8162448da83b083c444db9690ce01754625330385e274a127d"
      },
      {
        "file": "groups/group_161_key_169_DBH76_26.pickle",
        "sha256": "990d49b296363ea26846e06348f360c924e30abefaf32fb3d57afd271f02c993"
      },
      {
        "file": "groups/group_162_key_170_DBH76_27.pickle",
        "sha256": "c20a4aee5ce051d5ca9f6d8c3da56b971d08efb7da867472849bbc5a0f84436e"
      },
      {
        "file": "groups/group_163_key_171_DBH76_28.pickle",
        "sha256": "9d9d96b6a5aafa52617ff690abd12a209d9952715fef67c10b1b9ef16163f1b8"
      },
      {
        "file": "groups/group_164_key_172_DBH76_29.pickle",
        "sha256": "f0e29bd84791a53e5704d63cff545de74443d36f195bdb9ab974a17b4974382f"
      },
      {
        "file": "groups/group_165_key_173_DBH76_30.pickle",
        "sha256": "f147f298efe8767b6d5894f5770bdbc48f049fa82053d97022f773cc34f008a4"
      },
      {
        "file": "groups/group_166_key_174_DBH76_31.pickle",
        "sha256": "18473ef79b6e13528f5d4fe853d2465cab85976f440a62711d34d6d70348d45f"
      },
      {
        "file": "groups/group_167_key_175_DBH76_32.pickle",
        "sha256": "44c9fd5483959958423ce13246babe54a6507e4b0ca1bb88aacf5aa5b471d53b"
      },
      {
        "file": "groups/group_168_key_176_DBH76_33.pickle",
        "sha256": "63b153c23f0054d964fd3d12970c0f624846adac3b97d93f144bdf81b3d52355"
      },
      {
        "file": "groups/group_169_key_177_DBH76_34.pickle",
        "sha256": "61f47d16a83c2ba4659615621c9be1be9aedda056adee3b82d829ce46b73ff75"
      },
      {
        "file": "groups/group_170_key_179_DBH76_36.pickle",
        "sha256": "f102343f934d571ed82859511be505dfb8b9ac36bca231815c7cb6d147faf73c"
      },
      {
        "file": "groups/group_171_key_180_DBH76_37.pickle",
        "sha256": "05fb4614a47b5a8334cd0b54eeb4ade86bfa3a0e238923ab9cdfb91564de44f1"
      },
      {
        "file": "groups/group_172_key_181_DBH76_38.pickle",
        "sha256": "5d297e5fe294aaf3b640e22eb20be79723f7ed714979bfbdde6ac3496db9b17d"
      },
      {
        "file": "groups/group_173_key_182_DBH76_39.pickle",
        "sha256": "7f5b1cf6f855c1a4a2d301d0e9002b9288bb5b8406c7e4e3ced3f1214a1b913a"
      },
      {
        "file": "groups/group_174_key_183_DBH76_40.pickle",
        "sha256": "86c84fe39db1f2c40ad79f52f0c0d931a71994f029e546dedf61ad4fe9d35618"
      },
      {
        "file": "groups/group_175_key_184_DBH76_41.pickle",
        "sha256": "a4c57e16ded884b172cd214f35bd62e722f28b5efc0b4f4472533f73b847e152"
      },
      {
        "file": "groups/group_176_key_187_DBH76_44.pickle",
        "sha256": "a96aaf992850a31635e5857c582d19a32d594c7c860936b193674ec9ae014194"
      },
      {
        "file": "groups/group_177_key_188_DBH76_45.pickle",
        "sha256": "bc9dd6d895c8376e7fb1d5fdff8946bbd70b3bc8cf4c244e2779e62e825717c5"
      },
      {
        "file": "groups/group_178_key_189_DBH76_46.pickle",
        "sha256": "e1e5f77b268320d8ef5277dcbf8e31ff67b865ab2da2623d76624e93f1553d0a"
      },
      {
        "file": "groups/group_179_key_190_DBH76_47.pickle",
        "sha256": "a8a28e272e9f6ca5fa0648f1f80903fcfdca82250169ddde94dc4d9a51b5229a"
      },
      {
        "file": "groups/group_180_key_191_DBH76_48.pickle",
        "sha256": "19d6df60645f2fa2d48f65d58d901540dc9b2b58d39a77fd0496be79249182b3"
      },
      {
        "file": "groups/group_181_key_192_DBH76_49.pickle",
        "sha256": "c83da96115ed2f3b216eedc066a05df8032df92a276273fb7341a403ffd9e7f1"
      },
      {
        "file": "groups/group_182_key_193_DBH76_50.pickle",
        "sha256": "977d5cf9b31a1d15cc9f2757a908d80c56ae11afa867a1c3a7575eec87afa9a3"
      },
      {
        "file": "groups/group_183_key_194_DBH76_51.pickle",
        "sha256": "e5778b98f7db11772255f53f307f786b51e1aa84d11061341c3df82a4bb3cc4c"
      },
      {
        "file": "groups/group_184_key_195_DBH76_52.pickle",
        "sha256": "45794e8dc560afc635b9d9f63af7a13a72f067956fc4431a478df4a26f098894"
      },
      {
        "file": "groups/group_185_key_196_DBH76_53.pickle",
        "sha256": "4b6d1b1caa4720bd7baf060b7ea51d472dd43b39ec69f45177c2c27af93b039b"
      },
      {
        "file": "groups/group_186_key_199_DBH76_56.pickle",
        "sha256": "9e32bffaf21d4318812980ab9b54a7a1ccf5dc04b7d94cc3a372d06eda10dd35"
      },
      {
        "file": "groups/group_187_key_200_DBH76_57.pickle",
        "sha256": "90a3ea92e86c8569ec01feb063745d43549b6b7334073324a088dc657f8d1cec"
      },
      {
        "file": "groups/group_188_key_201_DBH76_58.pickle",
        "sha256": "ebd79e597d235fd30f7de5a190827827a969f02f98be4cc081bd7169a99826f0"
      },
      {
        "file": "groups/group_189_key_202_DBH76_59.pickle",
        "sha256": "a9854a15703e751fd4a8b04bf21fe718f7a3ed440553e78089caa0ce14a5d4e1"
      },
      {
        "file": "groups/group_190_key_203_DBH76_60.pickle",
        "sha256": "6ea089c6f4d3f9708c12b17449f9fd60c6d10765ec32c3dbf9aa90ef4d51f9ee"
      },
      {
        "file": "groups/group_191_key_204_DBH76_61.pickle",
        "sha256": "949ea1b830b7e944e6d94f6c11d84850f7f26be8277505741c514808cc8da233"
      },
      {
        "file": "groups/group_192_key_205_DBH76_62.pickle",
        "sha256": "63627c1d6f0a13fac27563154d17bf8b384fe3346d7488dc98e80eb3808bcb81"
      },
      {
        "file": "groups/group_193_key_206_DBH76_63.pickle",
        "sha256": "99a234a6b9992cc8358e2f5c5e98f184cadc8313de87d6954f479de611d2497c"
      },
      {
        "file": "groups/group_194_key_207_DBH76_64.pickle",
        "sha256": "17aca9c010a7e2a4ba29cc04a6bce46ffd780379cc705de4bab57f2dafd1101b"
      },
      {
        "file": "groups/group_195_key_208_DBH76_65.pickle",
        "sha256": "ca538d46b9905ff39f1664e4276c0934fc7977a4ec1d1a57e60a976e5af6a60f"
      },
      {
        "file": "groups/group_196_key_209_DBH76_66.pickle",
        "sha256": "65f8b7954150eece33c359ac13c29da740c1e8c94ed14cbf6f75201ca02fae6e"
      },
      {
        "file": "groups/group_197_key_210_DBH76_67.pickle",
        "sha256": "514b954c0df309528af4f299fe5d719f9f944868001da4a6b89fba23aab5d59f"
      },
      {
        "file": "groups/group_198_key_211_DBH76_68.pickle",
        "sha256": "2c5e4e190b5676eb892a76b71c3d57dee579692745eede359acff09fa6b9330c"
      },
      {
        "file": "groups/group_199_key_212_DBH76_69.pickle",
        "sha256": "c485e477b64a4f9ce04dd813c5fec09748d97cec04c20acd5756723e3fda9867"
      },
      {
        "file": "groups/group_200_key_213_DBH76_70.pickle",
        "sha256": "0dd6ae61e36009b622714085e8497bd1f46f5f11f23ba317e38ca1cf0cc2dc70"
      },
      {
        "file": "groups/group_201_key_214_DBH76_71.pickle",
        "sha256": "acd4c9ccf1ee8788802c778e72aa8b9ccc6d2e8d267aa1e450292dd3b5eaa84b"
      },
      {
        "file": "groups/group_202_key_215_DBH76_72.pickle",
        "sha256": "07f842e7847a8549eaada28233c1dafde30875486826003cf29d270df14d3e5a"
      },
      {
        "file": "groups/group_203_key_216_DBH76_73.pickle",
        "sha256": "8e89c92131952e6436bc80aeb0411c5b7f2af7cee3948125018f1aa0ab81bedf"
      },
      {
        "file": "groups/group_204_key_217_DBH76_74.pickle",
        "sha256": "485046cdb104f2ecb6f0c425e6bfbc4973dfb3815dfcef8265a9c6619bcf4f5a"
      },
      {
        "file": "groups/group_205_key_218_DBH76_75.pickle",
        "sha256": "8880571d68745fa9a2185e25e512f51f19c8d320a2d2649f5b9c4d4e7f59ff29"
      },
      {
        "file": "groups/group_117_key_122_EA13_0.pickle",
        "sha256": "bbcaf3fe1df00a00b1116e0aa9ade8ae994bad5486834311cfdeb7488b3e30ae"
      },
      {
        "file": "groups/group_118_key_123_EA13_1.pickle",
        "sha256": "263b527753d1a4715f398cf834b4e996dc8268e9cce43a8c1374af1b34574f6e"
      },
      {
        "file": "groups/group_119_key_124_EA13_2.pickle",
        "sha256": "6178c472b3bdfa57b3d3c05218cfb1fd8071d8b241a1edcee0447b9b952eef06"
      },
      {
        "file": "groups/group_120_key_125_EA13_3.pickle",
        "sha256": "e789a95a51b2b09f14519cd2d3f982ccd948fa2af8ba9374bfd2ef0ca00107ac"
      },
      {
        "file": "groups/group_121_key_127_EA13_5.pickle",
        "sha256": "91b0c68f200c29c009943bd7eb1f798c1e9c464445e08d1003a857f6ef138e3e"
      },
      {
        "file": "groups/group_122_key_128_EA13_6.pickle",
        "sha256": "c14f27e84bf317743081427b924579a4777c042050a4885fed15a34507460106"
      },
      {
        "file": "groups/group_123_key_129_EA13_7.pickle",
        "sha256": "326e905d4b28e1b487a1a8798680dc7c377507b99ae0c9f95f95b08a7b318501"
      },
      {
        "file": "groups/group_124_key_131_EA13_9.pickle",
        "sha256": "fb7dce677c13e240c939c4df55176410a3f5c6c71b41f40752164a00a9f39010"
      },
      {
        "file": "groups/group_125_key_132_EA13_10.pickle",
        "sha256": "ae6865d0d1ee0c45c61b2291225ecc6f698751405aad542c4559a48ec4ed73a6"
      },
      {
        "file": "groups/group_126_key_133_EA13_11.pickle",
        "sha256": "3e58e40c4a11007be8645233a7be9366ee20fd5ea227fd852a3e776711cdcd73"
      },
      {
        "file": "groups/group_127_key_134_EA13_12.pickle",
        "sha256": "ca8daebf808ade5121c9c294ff2ea7f3d3a81aa7f809e0a48f1d66626b8cf1af"
      },
      {
        "file": "groups/group_104_key_109_IP13_0.pickle",
        "sha256": "95ed47d36ae2d94bb6f27c7af6caaadc2e8f05a6c424b783b8a47edd61e03511"
      },
      {
        "file": "groups/group_105_key_110_IP13_1.pickle",
        "sha256": "1af643a9e18841fcec34dd6421f5ef9f0272470dc9db25518c287ae064510eab"
      },
      {
        "file": "groups/group_106_key_111_IP13_2.pickle",
        "sha256": "bed735bb15e9c48135eb11472d6aaf2bea297306256b4c6cbf3bf9ab38b31773"
      },
      {
        "file": "groups/group_107_key_112_IP13_3.pickle",
        "sha256": "7044399e68da7055fc0894906660da8a6e5d4700ec2590a6700dc8f3153480c1"
      },
      {
        "file": "groups/group_108_key_113_IP13_4.pickle",
        "sha256": "0fd84fd7197445372c04cfbe7d13ec39d057ae0d3c9d1f40c3ed47484a4c08b2"
      },
      {
        "file": "groups/group_109_key_114_IP13_5.pickle",
        "sha256": "a6dbcfe17b84002db702f5b39017cd650095989d19020781ec1c516e41ebb715"
      },
      {
        "file": "groups/group_110_key_115_IP13_6.pickle",
        "sha256": "ae7de875c168d75ff0e7f0ced47e42c388505bd1f9ea33ef33d511861c9bd11f"
      },
      {
        "file": "groups/group_111_key_116_IP13_7.pickle",
        "sha256": "5e3ccbff8b3f419590599dbf07624eb4540b707076bf2402f7b04b1498ff732f"
      },
      {
        "file": "groups/group_112_key_117_IP13_8.pickle",
        "sha256": "d52aa5f43fdd587fc7560484d7f743d098cc2c2292eca03f4fcb548a48955cbe"
      },
      {
        "file": "groups/group_113_key_118_IP13_9.pickle",
        "sha256": "ed0f5d56ae40656e32de2c58d4a0a9faedcbf3d2556ac6acbb67cdbdac384151"
      },
      {
        "file": "groups/group_114_key_119_IP13_10.pickle",
        "sha256": "b5a9a575c3c6418dd224a312b23a81fa8d2e3b95e5170552d980a4fb8c394c9a"
      },
      {
        "file": "groups/group_115_key_120_IP13_11.pickle",
        "sha256": "dd6bd45bb2ce19df577338a8595b90655d68c3b98916a1e3a0e05974923eb171"
      },
      {
        "file": "groups/group_116_key_121_IP13_12.pickle",
        "sha256": "41544090e9928eed2cfb657cd88d944682c68588e16780b3169e30971d680038"
      },
      {
        "file": "groups/group_000_key_000_MGAE109_0.pickle",
        "sha256": "d009b877f0c0f92fa5430ea9eb759e2ea9c89a090df41d5a97a8de686c4766cb"
      },
      {
        "file": "groups/group_001_key_001_MGAE109_1.pickle",
        "sha256": "a2971eb93bd3060f7a45b14d601332efed8e7de83a5b92961008d0c1139e1c4d"
      },
      {
        "file": "groups/group_002_key_002_MGAE109_2.pickle",
        "sha256": "f1d21b1694c9030122c05200de7566a874cfe73eb4430bdcab789066946bd7aa"
      },
      {
        "file": "groups/group_003_key_003_MGAE109_3.pickle",
        "sha256": "3bd990fcdaf66028c4a1417e9336d7dda95e24a5e59c066d5c617bbd8206cdc3"
      },
      {
        "file": "groups/group_004_key_004_MGAE109_4.pickle",
        "sha256": "271118569a12b1ec282c5d9fd777615e8e49bc2a53c65af8f79c10f5dc3bf873"
      },
      {
        "file": "groups/group_005_key_005_MGAE109_5.pickle",
        "sha256": "b9444fad91c2e546b2048c1b65a7c0a9d61c44933e441760d8cbcb84434af9ff"
      },
      {
        "file": "groups/group_006_key_006_MGAE109_6.pickle",
        "sha256": "2dd2c05e02e8cc1eb1498f49da6ae7432a9e3f8be8b2b1ebc4abafddda77a83a"
      },
      {
        "file": "groups/group_007_key_007_MGAE109_7.pickle",
        "sha256": "9cda8f2fb08343bf2f03fa0bab250cd69f14537f8197d720c4f5e628c21d353f"
      },
      {
        "file": "groups/group_008_key_008_MGAE109_8.pickle",
        "sha256": "b82a98719e7912dc21610c96acb7881a200728d0c7cf533938f51a0af2fd805a"
      },
      {
        "file": "groups/group_009_key_009_MGAE109_9.pickle",
        "sha256": "50d235db2321023f9c5990aec0768f805d11ada1e27aec562cb3cf4abff63ce5"
      },
      {
        "file": "groups/group_010_key_010_MGAE109_10.pickle",
        "sha256": "816475862c42d4eb67d9bedf09ac6c021a90da83c5d14dbef902d092d0712384"
      },
      {
        "file": "groups/group_011_key_011_MGAE109_11.pickle",
        "sha256": "22b12ff8f56409999afdba5d6371406f218817775f838ff8f58c6844232e18b2"
      },
      {
        "file": "groups/group_012_key_012_MGAE109_12.pickle",
        "sha256": "dbbbf3937a9040989632c36208e398c9685e2b1a99b474b3ea9b797f40781e51"
      },
      {
        "file": "groups/group_013_key_013_MGAE109_13.pickle",
        "sha256": "ee7e1aaacb54afbc84d13fe5eacf3edd5f9a39e15b7073ff6fb91670dd7411f4"
      },
      {
        "file": "groups/group_014_key_014_MGAE109_14.pickle",
        "sha256": "24efb9c122c0438eaa749b91890891d1795e3102717c61fefbe9fff0e9a4ba71"
      },
      {
        "file": "groups/group_015_key_015_MGAE109_15.pickle",
        "sha256": "cd15173882cd3c4c3608534abd87f821ee103e17449308ec33d7263899a92877"
      },
      {
        "file": "groups/group_016_key_016_MGAE109_16.pickle",
        "sha256": "f6eac98261e2b551b66a5db699fd31b9860b2fe82746c027c79b1377ba00509b"
      },
      {
        "file": "groups/group_017_key_017_MGAE109_17.pickle",
        "sha256": "ee874d516a1946b74036b68e8f95855f9e71bea3c0ad507fb417743e80b253d1"
      },
      {
        "file": "groups/group_018_key_019_MGAE109_19.pickle",
        "sha256": "24cc7d6386e4b80ba1bfbde7f3d99dce56dc5a9b795fd833efd45f1e9cde865a"
      },
      {
        "file": "groups/group_019_key_020_MGAE109_20.pickle",
        "sha256": "73b0327d8603162cc17f14598bd6a91ea857fafbb04dae650986366d04245417"
      },
      {
        "file": "groups/group_020_key_021_MGAE109_21.pickle",
        "sha256": "2d3ad5d10f2e69aae759e251a76b315941da9b85b33633539d860460b6554003"
      },
      {
        "file": "groups/group_021_key_022_MGAE109_22.pickle",
        "sha256": "e1fd092f28a729a6fd472fee1aa6cbe712f5dd1338e4c69a74d9fc6296018a5a"
      },
      {
        "file": "groups/group_022_key_023_MGAE109_23.pickle",
        "sha256": "3c33a70e7bff3c87c22ad958361000301419062e771d4a1555bdc6dfba9840a8"
      },
      {
        "file": "groups/group_023_key_024_MGAE109_24.pickle",
        "sha256": "e775746aa36d9b6a21b5450f85453fcd351f10402d30c07603471e9e6b43b006"
      },
      {
        "file": "groups/group_024_key_025_MGAE109_25.pickle",
        "sha256": "77645813f68523c89027b39092c9afc1461f5a822d0c0018e1ff6eae6f1bbce1"
      },
      {
        "file": "groups/group_025_key_026_MGAE109_26.pickle",
        "sha256": "6595d43a5d3b6e05861b62a065bc6331b507d4641bab9a1e82f34e55efd4c819"
      },
      {
        "file": "groups/group_026_key_027_MGAE109_27.pickle",
        "sha256": "dbb8c163490f83daaaaacb2f65282390095c00f3a5c811d4edbe6c0d31240e75"
      },
      {
        "file": "groups/group_027_key_029_MGAE109_29.pickle",
        "sha256": "3b47d5369d30d27769ec239f9739cd1ae4f730f9b06ea907ec7b94e71c79b4d8"
      },
      {
        "file": "groups/group_028_key_030_MGAE109_30.pickle",
        "sha256": "64639011aa6f59cdbf1cbbede15bb0865ee2c654cf89d3e68ab58e40eac1906f"
      },
      {
        "file": "groups/group_029_key_031_MGAE109_31.pickle",
        "sha256": "0903b593153a8e0a3ceefd72abc3c3b28eeb724a778408d896a08cf5a6bc958f"
      },
      {
        "file": "groups/group_030_key_032_MGAE109_32.pickle",
        "sha256": "5d6c14d6e2b6aeb2040ec79f494d3d57f765863053e02dac34c495290abab65d"
      },
      {
        "file": "groups/group_031_key_033_MGAE109_33.pickle",
        "sha256": "e686981909b53581ecd47539bc8f92fc810441a533452faa80b1a9aafd427d81"
      },
      {
        "file": "groups/group_032_key_035_MGAE109_35.pickle",
        "sha256": "33d144924435d79feaa1db3899a4408bef4f77528bcd6b113e63b32ea225f91e"
      },
      {
        "file": "groups/group_033_key_036_MGAE109_36.pickle",
        "sha256": "7ca6a6b30953e7dbc940deec12f425e98d6890a4d675cc8de250b57b0f1f784b"
      },
      {
        "file": "groups/group_034_key_037_MGAE109_37.pickle",
        "sha256": "7edc39d7e9e747a7cf02936e478b1de8e58d1d5f28ece40be1d77e1a3869d335"
      },
      {
        "file": "groups/group_035_key_038_MGAE109_38.pickle",
        "sha256": "851b0c4f99a95759e8e89e207da00efc444ae4d410aefd4ae3f3eb37f0ef155b"
      },
      {
        "file": "groups/group_036_key_039_MGAE109_39.pickle",
        "sha256": "473a2d1b9475af86fc230a8bebaf0e2cdfd067210bc2735705b4be42abc38139"
      },
      {
        "file": "groups/group_037_key_040_MGAE109_40.pickle",
        "sha256": "792d77cac9db3db6a00a5b9ba8f0d2108b65632a0cb9c3d262499041e9253091"
      },
      {
        "file": "groups/group_038_key_041_MGAE109_41.pickle",
        "sha256": "843b59c2e2c9621bc7ffc9af0ef0ca44c4155289556d0ca5835dd711e149f61b"
      },
      {
        "file": "groups/group_039_key_042_MGAE109_42.pickle",
        "sha256": "b0b7350a9a237a9ac22479d784030a0995ab87846f53ec6fdd66825bb5f9f690"
      },
      {
        "file": "groups/group_040_key_043_MGAE109_43.pickle",
        "sha256": "64c423177694a22bc11a8f88c6a6c3623ff34168125a09ef828f1bece8261c6e"
      },
      {
        "file": "groups/group_041_key_044_MGAE109_44.pickle",
        "sha256": "858aca6a90bf02e0bb9625e7da473296a3b690ed3aa91e54bd3639b1d6937808"
      },
      {
        "file": "groups/group_042_key_045_MGAE109_45.pickle",
        "sha256": "d9058c2c622b1d38d175a419dc9b3c3d431078282a8779c82de38b3e4b987165"
      },
      {
        "file": "groups/group_043_key_046_MGAE109_46.pickle",
        "sha256": "3cecbd98e183dab5ddf485fdf8a0b9c0b4bee971ce13bbde77d47e5a7834cfe1"
      },
      {
        "file": "groups/group_044_key_047_MGAE109_47.pickle",
        "sha256": "4be298800c1b980381d1a778f47399e00068c18aac91a49b362ad5cbfe14f919"
      },
      {
        "file": "groups/group_045_key_048_MGAE109_48.pickle",
        "sha256": "03bb4c3a12eb1e9387723d4a5b6c5936bb3575d8f6826adb16ee9b62494b0608"
      },
      {
        "file": "groups/group_046_key_049_MGAE109_49.pickle",
        "sha256": "ed0542e15c3b7f4f49f82199c1cbe72378fa8485e620bb25d59bc3af8b872dd7"
      },
      {
        "file": "groups/group_047_key_050_MGAE109_50.pickle",
        "sha256": "6b787ddeb34eb1bfe401c268d2ebbe555ba0e9ccf6244e6602f20202bfce89bc"
      },
      {
        "file": "groups/group_048_key_051_MGAE109_51.pickle",
        "sha256": "07cb718dd8d0d818e267753b1bda64a1940381a4f3a67e9468c77ab4fc7a387f"
      },
      {
        "file": "groups/group_049_key_052_MGAE109_52.pickle",
        "sha256": "715424c902dc2dfad7f4fe48bdf44e1a4b19afa7d21f334d8b4ab30f66b71994"
      },
      {
        "file": "groups/group_050_key_053_MGAE109_53.pickle",
        "sha256": "45c5f7eeda3508e53369a9aeceaf3116b877f2b6e00424ce72e40d8a7531f1e0"
      },
      {
        "file": "groups/group_051_key_054_MGAE109_54.pickle",
        "sha256": "7aa8a287ea1b6c9374ce8e8444579026372d999756bea5fd7674f08ed80f0b55"
      },
      {
        "file": "groups/group_052_key_056_MGAE109_56.pickle",
        "sha256": "b52d7b213bd387d5dd86c056b56b6f7efc45245d5be62c83d88569b45e90ec13"
      },
      {
        "file": "groups/group_053_key_057_MGAE109_57.pickle",
        "sha256": "6046c254dd878427e1214baf5d11e4b8b248fa94551fd1b46466b33874439acb"
      },
      {
        "file": "groups/group_054_key_058_MGAE109_58.pickle",
        "sha256": "b1f48f8939bd864156362ca28fafced70eaab200e4dde83c37a32bed7c8a1d4b"
      },
      {
        "file": "groups/group_055_key_059_MGAE109_59.pickle",
        "sha256": "e3f75d78ee9aed92b3304147b9ce8ff66e62a994c315d8e658499681e016d10f"
      },
      {
        "file": "groups/group_056_key_060_MGAE109_60.pickle",
        "sha256": "f4eab4fd37e29466f448a1a4e0a8360c18340ddd599de44cc14e71692760d648"
      },
      {
        "file": "groups/group_057_key_061_MGAE109_61.pickle",
        "sha256": "7f4e6be331e4c379fa90249128039c249366f092455fa2ca6666db9fe3a4cb54"
      },
      {
        "file": "groups/group_058_key_062_MGAE109_62.pickle",
        "sha256": "5a262d07734a307f994af555a00fbb281c0261b96808889b04325e1d9a2e016c"
      },
      {
        "file": "groups/group_059_key_063_MGAE109_63.pickle",
        "sha256": "8f5e5d50ea7d23bb88db61978d120bb0625f94da4bbee8f15b373026debd0998"
      },
      {
        "file": "groups/group_060_key_064_MGAE109_64.pickle",
        "sha256": "34a8cd4f5ad86940604eccfd02ead45261687e575ec7ba39773c2a11cc0d6330"
      },
      {
        "file": "groups/group_061_key_065_MGAE109_65.pickle",
        "sha256": "ca1a540c4cc9b3a7a1c4e1ec720c67255acaaf0358688313467b6cc017e67633"
      },
      {
        "file": "groups/group_062_key_066_MGAE109_66.pickle",
        "sha256": "76c6be1c635da58c6a108750d13157811e093b664f7b755eec49fe806302f049"
      },
      {
        "file": "groups/group_063_key_067_MGAE109_67.pickle",
        "sha256": "5f9085f114adcbb0eccb09168400ea91042b9b244fa1495dbfe8c53b80fcf2e2"
      },
      {
        "file": "groups/group_064_key_068_MGAE109_68.pickle",
        "sha256": "f5870daa8ffb02bb567e5750bc76f132b2ba9192c934411ac3af8794e5e38f11"
      },
      {
        "file": "groups/group_065_key_069_MGAE109_69.pickle",
        "sha256": "5e50f5e286f1dbeb6ed0c097b7eeb59b54bf08ecf19dd22c09629f76c208bcfe"
      },
      {
        "file": "groups/group_066_key_070_MGAE109_70.pickle",
        "sha256": "c6a52d3c6e5a41cccf16797cf444d807590568ae041f851b103997ba749dcd77"
      },
      {
        "file": "groups/group_067_key_071_MGAE109_71.pickle",
        "sha256": "431de51efe622956511ec976090e7d1195cf07f99ed3f7e612b737cd3c3dec3f"
      },
      {
        "file": "groups/group_068_key_072_MGAE109_72.pickle",
        "sha256": "7b020a344a310751857013dfb785721c86a5fbb20d5c50f1403c3de726e93b2d"
      },
      {
        "file": "groups/group_069_key_073_MGAE109_73.pickle",
        "sha256": "2e449c45ade2b9c336074bfb2c718dc9acdbbe27fd44a04ade773b040ce2bc36"
      },
      {
        "file": "groups/group_070_key_075_MGAE109_75.pickle",
        "sha256": "623bb69ee3104b5760d58a342fba1370382e5c931c1d940727a11d8d1f616df9"
      },
      {
        "file": "groups/group_071_key_076_MGAE109_76.pickle",
        "sha256": "b81652fdaab37f6f987e47823d7d31097f9b3b66ca6a46fe16fd47719bb0732b"
      },
      {
        "file": "groups/group_072_key_077_MGAE109_77.pickle",
        "sha256": "be7781512143c0abd2269b47b47a39989c521c8f3422e6c17fa52776cb3295ce"
      },
      {
        "file": "groups/group_073_key_078_MGAE109_78.pickle",
        "sha256": "2411fbe52b58b13395fe31caf7501c632d71a369eae26b3674fcb9801b6511b6"
      },
      {
        "file": "groups/group_074_key_079_MGAE109_79.pickle",
        "sha256": "5cc27b8d81f352caffa85b231c274d3435cf3a413d43e44ff21f9b8281f860c2"
      },
      {
        "file": "groups/group_075_key_080_MGAE109_80.pickle",
        "sha256": "c015549989238a24b04a3b61a4ecbd9b829a544562d1f706e94dffaab01e1dc5"
      },
      {
        "file": "groups/group_076_key_081_MGAE109_81.pickle",
        "sha256": "5e953aa971875d346d671703922de78ab05030172f8c11b4174159c03cb65caf"
      },
      {
        "file": "groups/group_077_key_082_MGAE109_82.pickle",
        "sha256": "bc9ca9499b37ded8f48e83f76b5cc7b51131edd0273c683a04a2a47ff3278805"
      },
      {
        "file": "groups/group_078_key_083_MGAE109_83.pickle",
        "sha256": "6e691d196e4763cf2df692e384684b80b1696fb92f31c594da638fec1c45e442"
      },
      {
        "file": "groups/group_079_key_084_MGAE109_84.pickle",
        "sha256": "2441cfca4a606c9daf690d7ec7590877e70b5e15f55a5414d8412b030ec6c141"
      },
      {
        "file": "groups/group_080_key_085_MGAE109_85.pickle",
        "sha256": "aea046fa3c63f039d0dc98904701c8e3bef7fe6fa6d1447a618d7b6128f01e3e"
      },
      {
        "file": "groups/group_081_key_086_MGAE109_86.pickle",
        "sha256": "bf567e077252fb4e929e6eb9f1daeea8847fc859ba5575c62e2bba13547993d1"
      },
      {
        "file": "groups/group_082_key_087_MGAE109_87.pickle",
        "sha256": "5a575ddf0424634e1d76d9818c3ceda756326796455b57fc9456e4d7c6e48173"
      },
      {
        "file": "groups/group_083_key_088_MGAE109_88.pickle",
        "sha256": "e899af9c9f72af777050ea77ca6c68d51fce43d144ec95665ae6993e90bc6827"
      },
      {
        "file": "groups/group_084_key_089_MGAE109_89.pickle",
        "sha256": "f94a124078e847eaf09407b80d1a74294533037a809f4129224b9f940ce81242"
      },
      {
        "file": "groups/group_085_key_090_MGAE109_90.pickle",
        "sha256": "9eed21255d667a39ce9b978437aec0816f34e02206ea2d9c3002959018567b3d"
      },
      {
        "file": "groups/group_086_key_091_MGAE109_91.pickle",
        "sha256": "f1589640ec71a6fdd28bb48b5fd7df0fde3a7d22a5501a11d78894b9221686fe"
      },
      {
        "file": "groups/group_087_key_092_MGAE109_92.pickle",
        "sha256": "fea1b9920384224319bc5d6b8d87e65125a361460e50bb9a90dc498922a6f86b"
      },
      {
        "file": "groups/group_088_key_093_MGAE109_93.pickle",
        "sha256": "68621035c254df786552b561d411954e948952232a5da045aa17053efd23583d"
      },
      {
        "file": "groups/group_089_key_094_MGAE109_94.pickle",
        "sha256": "2ff804af3e1e511900fa053b45e2a8199489b965f57c440a7c60ef7b53cfb02a"
      },
      {
        "file": "groups/group_090_key_095_MGAE109_95.pickle",
        "sha256": "91ae60a646a528105c0b0eaab6cfdd8c5f9c1cb004f539e3cb469116bd86ea40"
      },
      {
        "file": "groups/group_091_key_096_MGAE109_96.pickle",
        "sha256": "bd3ecb8d8e7968cbcbfbdfc6d1e92075f184001ffc279da1ef454a89a3b4cbef"
      },
      {
        "file": "groups/group_092_key_097_MGAE109_97.pickle",
        "sha256": "95de40d18b133ba1e78123849af9a3cd136d6e282a6d133d466a27f70aef2a4c"
      },
      {
        "file": "groups/group_093_key_098_MGAE109_98.pickle",
        "sha256": "bd640d82f5a3aaed9253342e383f64843c86d32a645e2b600cd07149e6183bdc"
      },
      {
        "file": "groups/group_094_key_099_MGAE109_99.pickle",
        "sha256": "7c73845becc8852cefd31d4230f94e2b735de26797a1ed9a5949761607afbbaa"
      },
      {
        "file": "groups/group_095_key_100_MGAE109_100.pickle",
        "sha256": "166534af46aa71e847dd8acdb53072fb315ff640a3ba999c27a7e3d0fd35f228"
      },
      {
        "file": "groups/group_096_key_101_MGAE109_101.pickle",
        "sha256": "ed5fc85f9f47043ef43102787f5735994b2d7c34aae856cc9329d579942d2cc0"
      },
      {
        "file": "groups/group_097_key_102_MGAE109_102.pickle",
        "sha256": "8dd6ec1bd54a33ccb2d20750ac127765a560786833e69e34a5ed1990c9ae8d2e"
      },
      {
        "file": "groups/group_098_key_103_MGAE109_103.pickle",
        "sha256": "ecda737e4aaa305f7a90ae9878db403b0ab809e945d15d74f263a34e0a2ff97c"
      },
      {
        "file": "groups/group_099_key_104_MGAE109_104.pickle",
        "sha256": "3c765a85e5ee2ca41ea022a718cb01a8439433edf9997c0a3267baf2eb18c896"
      },
      {
        "file": "groups/group_100_key_105_MGAE109_105.pickle",
        "sha256": "ecfe168df4bb974af45739e1ae4425e7e13be3262b607b2814a57ba752c89b4e"
      },
      {
        "file": "groups/group_101_key_106_MGAE109_106.pickle",
        "sha256": "73e69b0aa09ecd8f0c8948388838806e55ab6c834a04211b15153a6e1b86dcd6"
      },
      {
        "file": "groups/group_102_key_107_MGAE109_107.pickle",
        "sha256": "211de450276977e34ded98b1c8199c55a6a1be6cc6f72f8cfb32b7ed7daefeb2"
      },
      {
        "file": "groups/group_103_key_108_MGAE109_108.pickle",
        "sha256": "89f9b663cc6eed9d2d31cbca83fb82830b3c57b780a50680b8dee619e5b1535b"
      },
      {
        "file": "groups/group_206_key_219_NCCE31_0.pickle",
        "sha256": "c90140e87c652e251f1e788b240f0bb3a073bfb6b2cda6d7efc184ab903d2fe4"
      },
      {
        "file": "groups/group_207_key_220_NCCE31_1.pickle",
        "sha256": "ec5db877163fe9f3eace24c15e9a788c2c378dc2518d10e1b1a4c6f69c065463"
      },
      {
        "file": "groups/group_208_key_221_NCCE31_2.pickle",
        "sha256": "356108e58004f5eb8ebe49b7cae291ca31cc91a66005a69ef0fb7707328c46f2"
      },
      {
        "file": "groups/group_209_key_222_NCCE31_3.pickle",
        "sha256": "8224141902deed8e901de66ee8222f6a57f0f4945e9cd5ef7ba2e18fdb28bcea"
      },
      {
        "file": "groups/group_210_key_223_NCCE31_4.pickle",
        "sha256": "0ff1465d3b36d4cf27fb0299a9b6a52caf88c83fde1ca68c19504d41452272d2"
      },
      {
        "file": "groups/group_211_key_224_NCCE31_5.pickle",
        "sha256": "1abd1d4d30c127ab1423bcfa7f7f5a9e4927fc8a28214aea8f9ef556e9170e30"
      },
      {
        "file": "groups/group_212_key_225_NCCE31_6.pickle",
        "sha256": "cef4f5ceeb1b3a7b97f9cd0bc1c0ddeaec3c70b1d63e08f033902872b79571b5"
      },
      {
        "file": "groups/group_213_key_226_NCCE31_7.pickle",
        "sha256": "8934bed47b1d26a0cd1d3d7004ed3db8f498ce854335e893ce84de08c3fac630"
      },
      {
        "file": "groups/group_214_key_227_NCCE31_8.pickle",
        "sha256": "fa14c2e39853256a6c07ac6c8151881b61fe71269d74116678d80a4cfc94391c"
      },
      {
        "file": "groups/group_215_key_228_NCCE31_9.pickle",
        "sha256": "78b0d485cb8e1fd9114163da77cd9d47c00efe8347e676ab09428b37fceccf54"
      },
      {
        "file": "groups/group_216_key_229_NCCE31_10.pickle",
        "sha256": "b00a4f9292a48d334f98368402b2da9fac494266949d8fc750a1e4d018247223"
      },
      {
        "file": "groups/group_217_key_230_NCCE31_11.pickle",
        "sha256": "7558cd3df911b5e01a101f36e7b84c04ee228e838baa6cd7491ce4659fd61617"
      },
      {
        "file": "groups/group_218_key_232_NCCE31_13.pickle",
        "sha256": "2cc49fd91cbcda6f7c8db544e5a7fd5f2612a8d3baa292c1332ea5041b6d1934"
      },
      {
        "file": "groups/group_219_key_233_NCCE31_14.pickle",
        "sha256": "cc8e12542084f5fb40dc6d05cef7dc3855ed04c1a6b5edc3906261b2430d1d27"
      },
      {
        "file": "groups/group_220_key_234_NCCE31_15.pickle",
        "sha256": "d46a05c3f064f9b4b083b0b4535805ebf1b958cb018028c22f7ce3829e079ddf"
      },
      {
        "file": "groups/group_221_key_235_NCCE31_16.pickle",
        "sha256": "58856fbff944c5c6bf455099a43c5aa25e0163cd6320d9656179f35cb91f2593"
      },
      {
        "file": "groups/group_222_key_236_NCCE31_17.pickle",
        "sha256": "6c23ce92a71f8977e130e349666a5ed385018909afae1f368e17cffa3fd7e461"
      },
      {
        "file": "groups/group_223_key_237_NCCE31_18.pickle",
        "sha256": "370700d818a9898277138ae2abb4e29b98440d57a34bd48ba1c115b5b47354c8"
      },
      {
        "file": "groups/group_224_key_238_NCCE31_19.pickle",
        "sha256": "9973a10fa85e0c56082a6543336054ea3f2a0f25f2a815c55192f9fc5006ad4b"
      },
      {
        "file": "groups/group_225_key_239_NCCE31_20.pickle",
        "sha256": "85a2c33d430f6d8b61deac44d25f2d365da847ba17f423d49976339b59aa2bca"
      },
      {
        "file": "groups/group_226_key_241_NCCE31_22.pickle",
        "sha256": "109e90b3a676b5879d3de913b964c3fbd3b820ba4ec9b9957634a108500a24f2"
      },
      {
        "file": "groups/group_227_key_242_NCCE31_23.pickle",
        "sha256": "638d4eecac3ae5655b3d1f0220a238bf7d31b189f7cdf0b0e6eab296e702f7cc"
      },
      {
        "file": "groups/group_228_key_243_NCCE31_24.pickle",
        "sha256": "ded35fb75c4981677c491f3d76b84ed13e4bdbcd9a5d17746866cd1ad8a8887b"
      },
      {
        "file": "groups/group_229_key_244_NCCE31_25.pickle",
        "sha256": "5d5abc78dcbd5428c248dd78410ffedba090dfd2021834806daf00373826bb87"
      },
      {
        "file": "groups/group_230_key_245_NCCE31_26.pickle",
        "sha256": "97b3b2201b9fc278be7353488b022b598d048de4ceebfc802fecca0593f6ccba"
      },
      {
        "file": "groups/group_231_key_246_NCCE31_27.pickle",
        "sha256": "937a8aeb63d196af1b18d85b76cd77b0fc424a965f3e50349913fde65b74735b"
      },
      {
        "file": "groups/group_232_key_247_NCCE31_28.pickle",
        "sha256": "fd5928603c88b05cf7741f7f68ec540f3822f824c119d463fe9de66371d05165"
      },
      {
        "file": "groups/group_233_key_248_NCCE31_29.pickle",
        "sha256": "8d62858a593a4913dc42b54475b7571cce625559a2c080dd83ed916d06347d40"
      },
      {
        "file": "groups/group_128_key_135_PA8_0.pickle",
        "sha256": "8c719be12ddea1336f198849897fa32bb9f63a97a7160be7e1fcbecfdae1cbd4"
      },
      {
        "file": "groups/group_129_key_136_PA8_1.pickle",
        "sha256": "ec435d05dd7c3680cea5b0d568f74af761c488b5f8b5019f911afe0f449819f3"
      },
      {
        "file": "groups/group_130_key_137_PA8_2.pickle",
        "sha256": "b3fe7d86d3827fb5916007c46f0e1bc17f6a9f4ff5681a405ee0671cbbf92524"
      },
      {
        "file": "groups/group_131_key_138_PA8_3.pickle",
        "sha256": "82619d678eb6c5ecea18db942df9c169eae9cd69bb1fdd4ec87d70007374a320"
      },
      {
        "file": "groups/group_132_key_139_PA8_4.pickle",
        "sha256": "32ebcd9c4a14b994e788d9e9dc6f3ab3f1a261bb1d4c175ff272d3276e262ab4"
      },
      {
        "file": "groups/group_133_key_140_PA8_5.pickle",
        "sha256": "5cfe112a64e5fe5e8b4f920d2cc389c26d51f369ecd5779665414516c08dc04a"
      },
      {
        "file": "groups/group_134_key_141_PA8_6.pickle",
        "sha256": "3e371a71fbd83f7f0db7e16ff57584b722993b28d85b25ae32811806374e670b"
      },
      {
        "file": "groups/group_135_key_142_PA8_7.pickle",
        "sha256": "8e1baa02c96bebb0aea709e45dd2e1bc083e2148692338533ead1be9e2dd7e5b"
      },
      {
        "file": "groups/group_255_key_271_pTC13_0.pickle",
        "sha256": "ec824fd2d370add55c0b3bfd399d8d6e284505642f3c12373186e629ca49e5be"
      },
      {
        "file": "groups/group_256_key_272_pTC13_1.pickle",
        "sha256": "2a3c7e824206ad6bfcc0c2124909814b540063cde53726c0642199ef8e668bab"
      },
      {
        "file": "groups/group_257_key_273_pTC13_2.pickle",
        "sha256": "d27759ebe2a2eed7870a8117771fdd1fe495aaf88bb55d69d1df65c39d21cb1e"
      },
      {
        "file": "groups/group_258_key_274_pTC13_3.pickle",
        "sha256": "89c01de4ee7a7fe9df374386251eb0743a502b4849f299f52fc58cd26cd3d682"
      },
      {
        "file": "groups/group_259_key_275_pTC13_4.pickle",
        "sha256": "0cf9209ce3226df0f564f8604287ce146d96a426a4fc140f13ef6e0f0d706dd9"
      },
      {
        "file": "groups/group_260_key_276_pTC13_5.pickle",
        "sha256": "4d713cd6949d3256807b1f981661ad37d3aac604b8c05d1a8711153bcb24f70a"
      },
      {
        "file": "groups/group_261_key_277_pTC13_6.pickle",
        "sha256": "c3e5385e0aeb1e59cb250d880e342f3f4f7186e072e9d2e3f36792ac8d57ac4d"
      },
      {
        "file": "groups/group_262_key_278_pTC13_7.pickle",
        "sha256": "e5d76004df7293b81c8d2fa1fa03892276dd41f72c7e8375ad11eed391226caa"
      },
      {
        "file": "groups/group_263_key_279_pTC13_8.pickle",
        "sha256": "9967644a72f48f020cfb0113160ec29391c84a9a7e81220e6eb0c077d5f30571"
      },
      {
        "file": "groups/group_264_key_280_pTC13_9.pickle",
        "sha256": "514ab515968270747ae2a0079b9635bc92fac68497e618595f76fa0efcb5cb6d"
      },
      {
        "file": "groups/group_265_key_281_pTC13_10.pickle",
        "sha256": "7992daaa9a08536d0af2db58370d017d201f22e9b3702ac619e504aa45db454f"
      },
      {
        "file": "groups/group_266_key_282_pTC13_11.pickle",
        "sha256": "571344685d8d1bfad75e24ad7bb66e38d817e56eb01055401a2ec8a50ceab75c"
      },
      {
        "file": "groups/group_267_key_283_pTC13_12.pickle",
        "sha256": "93341724fbfb56aeba2951c0804769392eb3ce77d03a94a775239c867c6e680e"
      }
    ],
    "species": 358,
    "species_grids": 2864,
    "variants": 2144
  },
  "mrks": {
    "has_exc": 90,
    "has_operator": 90,
    "has_pointwise_vxc": 90,
    "systems": 90
  },
  "validation": {
    "clean_reactions": 28,
    "diagnostic_reactions": 30,
    "fixed_density_parity": "PASS",
    "primary_dispersion_values": 84,
    "qualification": {
      "component_density_descriptor_xc_exact": true,
      "component_max_abs_error_hartree": 1.1368683772161603e-12,
      "component_receipt_sha256": "5a64f01fa941d2e18cca8558b141dd4edc7fa707caa87082f872009a9309c441",
      "component_species": 2,
      "component_systems": [
        "MCONF-1-1",
        "MCONF-1-2"
      ],
      "description": "82 direct fixed-density total-energy parity + 2 large-system component-level parity",
      "direct_max_abs_error_hartree": 5.684341886080801e-13,
      "direct_species": 82,
      "qualified_species": 84,
      "status": "PASS"
    },
    "secondary_dispersion_values": 84,
    "species": 84
  }
}
```

268 chemistry identities: 251 relchem and 17 AE17. Eight augmentation variants per reaction (2144 reaction/variant combinations), never eightfold sample weight.
All 90 mRKS systems have energy, gauge-fixed pointwise and weak-form AO targets and factors.
Diet diagnostic: 30, nonselectable. Clean validation: 28. PBE0 species: 84.

## Exclusions and split integrity

Training exclusions (unchanged):

```json
{
  "DBH76": [
    15,
    35,
    42,
    43,
    54,
    55
  ],
  "EA13": [
    4,
    8
  ],
  "MGAE109": [
    18,
    28,
    34,
    55,
    74
  ],
  "NCCE31": [
    12,
    21,
    30
  ]
}
```

Validation exclusions:

| Validation | Counterpart | Reason |
|---|---|---|
| BH76-5 | Diet100:BH76-6 | identical stoichiometry + charge/spin + geometry fingerprint |
| G21EA-25 | Diet100:G21EA-25 | identical stoichiometry + charge/spin + geometry fingerprint |
| BH76-5 | Minnesota:DBH76-0 | composition/charge/spin/stoichiometry collision; differing geometries do not prove distinct chemistry; conservatively withheld |

Unique excluded identities: BH76-5, G21EA-25. 30 - 2 = 28.
BH76-5 has both reserved-test overlap and a conservative ambiguous training-collision flag; it is counted only once.

84/84 validation species qualified: 82 direct fixed-density total-energy parity + 2 large-system component-level parity.
Direct maximum discrepancy: 5.6843418860808015e-13 Ha. Component maximum: 1.1368683772161603e-12 Ha.
Both MCONF records have exact source-density, descriptor and PBE XC equality. Independent non-XC/recombined errors are 1.1368683772161603e-12 and 4.547473508864641e-13 Ha.

No future test split/arrays/labels are included. Reserved Diet100 was consulted only for leakage identities.

Primary dispersion: **PBE0-D3(BJ)**. Secondary: PBE-D3(BJ), never chosen by score.

## Units and precision

| Field | Axes | Units |
|---|---|---|
| Densities | point,spin | bohr^-3 |
| DensityDescriptorsN10 | point,rho2_grad6_lapl2 | rho bohr^-3; grad bohr^-4; lapl bohr^-5 |
| Exc | scalar | hartree |
| Gradients | point,sigma_aa_ab_bb | bohr^-8 |
| Grid | point,descriptor | mixed: rho bohr^-3; sigma bohr^-8; tau/lapl bohr^-5 |
| HF_energies | component | hartree |
| Overlap | ao,ao | dimensionless |
| PBE_local_energies | point | hartree/electron |
| RefAO | ao,ao | hartree |
| VxcLegacy | point | hartree |
| Weights | point | bohr^3 |
| coords64 | point,xyz | bohr |
| dm | ao,ao or spin,ao,ao | electron |
| dmks | ao,ao | electron |
| features | point,rho2_grad6_lapl2 | rho bohr^-3; grad bohr^-4; lapl bohr^-5 |
| fixed_nonxc | scalar | hartree |
| grad_phi | point,xyz,ao | bohr^-5/2 |
| lap_phi | point,ao | bohr^-7/2 |
| legacycoords32 | point,xyz | bohr |
| nonxc | scalar | hartree |
| npzrow | point | index |
| pbe_epsilon | point | hartree/electron |
| phi | point,ao | bohr^-3/2 |
| sourcerow | point | index |
| weights | point | bohr^3 |

Source/storage/production dtypes are explicit attributes on every numerical dataset. Chemistry retains F32 source rounding before matched-F64 arithmetic; operator precision boundaries are unchanged; validation density matrices/descriptors remain F64.
The non-XC validation term is Tr(P hcore) + 0.5 Tr(P J[P]) + E_nuc. No SCF iterations or benchmark scoring were run.

## Legacy and fixed-density parity

```json
{
  "chemistry": {
    "arrays_energy_loss_gradient": "exact",
    "identity_weighting": "268 identities; eight augmentations",
    "reaction_variant_cases": 72,
    "status": "PASS"
  },
  "mrks": {
    "all_system_input_parity": [
      "AlBeH",
      "AlBeH_iso2",
      "AlCl",
      "AlF",
      "AlH",
      "AlH3",
      "AlHMg",
      "AlHO",
      "AlHO_iso2",
      "AlHS",
      "AlHS_iso2",
      "BCl",
      "BF",
      "BFH2",
      "BH",
      "BH2Li",
      "BH2N",
      "BH2N_iso2",
      "BH3",
      "BHMg",
      "BHO",
      "BHS",
      "BHS_iso2",
      "Be2H2",
      "BeClH",
      "BeFH",
      "BeH2",
      "BeHLi",
      "BeHN",
      "BeHNa",
      "C2H2",
      "C2H2_iso2",
      "CClH",
      "CFH",
      "CH2",
      "CH2O",
      "CH2O_iso2",
      "CH4",
      "CHN",
      "CHN_iso2",
      "CHP",
      "CO",
      "CS",
      "ClH",
      "ClHMg",
      "ClHO",
      "ClHS",
      "ClHSi",
      "ClLi",
      "ClNa",
      "FH",
      "FH2N",
      "FHMg",
      "FHO",
      "FHS",
      "FHSi",
      "FLi",
      "FNa",
      "H2",
      "H2LiN",
      "H2Mg",
      "H2N2",
      "H2N2_iso2",
      "H2O",
      "H2O2",
      "H2S",
      "H2Si",
      "H3N",
      "H3P",
      "H4Si",
      "HLi",
      "HLiMg",
      "HLiO",
      "HLiS",
      "HMgNa",
      "HNO",
      "HNS",
      "HNS_iso2",
      "HNSi",
      "HNa",
      "HNaO",
      "HNaS",
      "HOP",
      "HPS",
      "HPSi",
      "HPSi_iso2",
      "LiNa",
      "N2",
      "OSi",
      "SSi"
    ],
    "comparison": "exact",
    "fresh_loss_gradient_parity": [
      "H2",
      "LiNa",
      "N2",
      "SSi"
    ],
    "historical_15_parity": "exact PASS",
    "historical_receipt_sha256": "45033c28f655eeaf4a79191e1eb68ed44e3d5d606ca7848e97f800d6f3c52379",
    "status": "PASS"
  },
  "validation": {
    "component_density_descriptor_xc_exact": true,
    "component_max_abs_error_hartree": 1.1368683772161603e-12,
    "component_receipt_sha256": "5a64f01fa941d2e18cca8558b141dd4edc7fa707caa87082f872009a9309c441",
    "component_species": 2,
    "component_systems": [
      "MCONF-1-1",
      "MCONF-1-2"
    ],
    "description": "82 direct fixed-density total-energy parity + 2 large-system component-level parity",
    "direct_max_abs_error_hartree": 5.684341886080801e-13,
    "direct_species": 82,
    "qualified_species": 84,
    "species": [
      "ACONF-10-H_ggg",
      "ACONF-10-H_ttt",
      "Amino20x4-28-GLU_xad",
      "Amino20x4-28-GLU_xbi",
      "Amino20x4-54-PHE_xar",
      "Amino20x4-54-PHE_xaw",
      "BH76-5-h",
      "BH76-5-hcl",
      "BH76-5-hclhts",
      "BHPERI-11-13_c2h4",
      "BHPERI-11-13r_1",
      "BHPERI-11-13ts_1a",
      "BHROT27-16-acetamide_RC",
      "BHROT27-16-acetamide_TS2",
      "BHROT27-26-ethylthiourea_180",
      "BHROT27-26-ethylthiourea_TS2",
      "BSR36-31-c2h6",
      "BSR36-31-ch4",
      "BSR36-31-r16",
      "BUT14DIOL-13-B1",
      "BUT14DIOL-13-B14",
      "CDIE20-9-P43",
      "CDIE20-9-R43",
      "DC13-1-ISO_E36",
      "DC13-1-ISO_P36",
      "DIPCS10-7-h2s",
      "DIPCS10-7-h2s_2+",
      "FH51-24-H2O",
      "FH51-24-butanediol",
      "FH51-24-dimethyloxirane",
      "FH51-30-C3H7NCO",
      "FH51-30-C3H7NH2",
      "FH51-30-CO2",
      "FH51-30-H2O",
      "G21EA-14-EA_14",
      "G21EA-14-EA_14n",
      "G21EA-25-EA_25",
      "G21EA-25-EA_25n",
      "HAL59-40-MeI",
      "HAL59-40-MeI_OPH3",
      "HAL59-40-OPH3",
      "HAL59-57-28_CH3I-benA",
      "HAL59-57-28_CH3I-benAB",
      "HAL59-57-28_CH3I-benB",
      "HEAVY28-16-h2o",
      "HEAVY28-16-sbh3",
      "HEAVY28-16-sbh3_h2o",
      "MB16-43-10-10",
      "MB16-43-10-BH3",
      "MB16-43-10-CH4",
      "MB16-43-10-Cl2",
      "MB16-43-10-F2",
      "MB16-43-10-H2",
      "MB16-43-10-MgH2",
      "MB16-43-10-N2",
      "MB16-43-10-SiH4",
      "MCONF-1-1",
      "MCONF-1-2",
      "PNICO23-16-16",
      "PNICO23-16-16a",
      "PNICO23-16-16b",
      "PX13-9-hf_2",
      "PX13-9-hf_2_ts",
      "S66-50-50",
      "S66-50-50A",
      "S66-50-50B",
      "S66-6-06A",
      "S66-6-06B",
      "S66-6-6",
      "SIE4x4-15-h2o",
      "SIE4x4-15-h2o+",
      "SIE4x4-15-h2o2+_1.5",
      "W4-11-132-cl",
      "W4-11-132-o",
      "W4-11-132-oclo",
      "W4-11-30-cl",
      "W4-11-30-h",
      "W4-11-30-hcl",
      "W4-11-57-b",
      "W4-11-57-bf",
      "W4-11-57-f",
      "WCPT18-15-h2o",
      "WCPT18-15-reac6",
      "WCPT18-15-ts6h2o"
    ],
    "status": "PASS"
  }
}
```

## Loader benchmark

```json
{
  "logical_sha256": "61c221a19b9987717e69cac182ad545241f8807db4126c0949a99992e4c210ef",
  "workloads": {
    "chemistry_random_variant": {
      "first_handle_access_seconds_including_integrity_check": 0.46172549999755574,
      "same_sample_warm_seconds": 0.0031822199991438536,
      "dataloader": {
        "0": {
          "first_epoch_items_per_second": 7.582268202353581,
          "warm_epoch_items_per_second": 110.26779222751522
        },
        "2": {
          "first_epoch_items_per_second": 3.4486673304019426,
          "warm_epoch_items_per_second": 163.4973722885794
        }
      },
      "sampled_peak_host_ram_bytes_parent_and_workers": null
    },
    "chemistry_sequential": {
      "first_handle_access_seconds_including_integrity_check": 0.1847295000043232,
      "same_sample_warm_seconds": 0.006169919999956619,
      "dataloader": {
        "0": {
          "first_epoch_items_per_second": 39.204914532243045,
          "warm_epoch_items_per_second": 127.93613427950868
        },
        "2": {
          "first_epoch_items_per_second": 5.243688458958709,
          "warm_epoch_items_per_second": 201.19358092694046
        }
      },
      "sampled_peak_host_ram_bytes_parent_and_workers": null
    },
    "mrks_energy": {
      "first_handle_access_seconds_including_integrity_check": 0.1846813000011025,
      "same_sample_warm_seconds": 0.01171852000115905,
      "dataloader": {
        "0": {
          "first_epoch_items_per_second": 6.2304551109916755,
          "warm_epoch_items_per_second": 84.97151329836699
        },
        "2": {
          "first_epoch_items_per_second": 3.6166795292377607,
          "warm_epoch_items_per_second": 143.3406616269226
        }
      },
      "sampled_peak_host_ram_bytes_parent_and_workers": null
    },
    "operator_chunks": {
      "first_handle_access_seconds_including_integrity_check": 0.20597080000152346,
      "same_sample_warm_seconds": 0.026927379998960534,
      "dataloader": {
        "0": {
          "first_epoch_items_per_second": 37.9280566403885,
          "warm_epoch_items_per_second": 37.93582633372639
        },
        "2": {
          "first_epoch_items_per_second": 5.364640481075299,
          "warm_epoch_items_per_second": 66.5091508286615
        }
      },
      "sampled_peak_host_ram_bytes_parent_and_workers": null
    },
    "validation_species": {
      "first_handle_access_seconds_including_integrity_check": 0.014515900002152193,
      "same_sample_warm_seconds": 0.007610619999468327,
      "dataloader": {
        "0": {
          "first_epoch_items_per_second": 43.30332953881724,
          "warm_epoch_items_per_second": 74.15585149978716
        },
        "2": {
          "first_epoch_items_per_second": 5.801216319263569,
          "warm_epoch_items_per_second": 134.47683468851824
        }
      },
      "sampled_peak_host_ram_bytes_parent_and_workers": null
    }
  },
  "method": "Fresh worker-local HDF5 handles; first access SHA checks included; OS cache not forcibly purged; persistent workers for second epoch",
  "no_model_evaluation": true
}
```

## Content and shard hashes

Logical SHA256: `61c221a19b9987717e69cac182ad545241f8807db4126c0949a99992e4c210ef`

Total dataset-file bytes (manifest inventory): 29691268078

| Shard | Bytes | SHA256 |
|---|---:|---|
| chemistry/chemistry_000.h5 | 297306184 | `24101423a5d61f4165800bc12d626ca62253ebd2e9e67d20ae7fd59a50d98ee6` |
| chemistry/chemistry_001.h5 | 367833913 | `b7f6ab8579512bf3b8cb4d47464642145314c820da5e795cba278365dc6ae41a` |
| chemistry/chemistry_002.h5 | 369097045 | `49762e9d01865038492f6c9ec7594557401957b84378d200bc217eeed216b12b` |
| chemistry/chemistry_003.h5 | 327965231 | `74015f0ea433fc4f76e8642ec38749fa243e9ada81e9aa54e4ed72509327fa5d` |
| chemistry/chemistry_004.h5 | 289777536 | `4ff7c1dc642bc55819ad141bcbc0272c76be0c126f0fccb248659f7266655a0a` |
| chemistry/chemistry_005.h5 | 289866193 | `88e2506d08e6ad40142e97c779f3521980e492489a2ff22e3b63dbe4d399f2dc` |
| chemistry/chemistry_006.h5 | 288739954 | `0770a88df0d01d8a4b72a7197fd912b12f8b43f6009ef2bbee3671fc5771785d` |
| chemistry/chemistry_007.h5 | 303703087 | `1fb84c1e110a7fa4becb69c283519b6605089751ef2f02138ba506887289ece5` |
| chemistry/chemistry_008.h5 | 305205653 | `f7199c031aeea96ba7b4526dd85085bdb1860b3babdf58e65890f088b770e8a2` |
| chemistry/chemistry_009.h5 | 357070114 | `af40fdd369db6314147fe3801e867ec0e3c634f60d33014adfd87c416ea41103` |
| chemistry/chemistry_010.h5 | 335325152 | `030e4f12c799a1684482a25f9276b19c6336c51d3095326af671b2652b1f1ed7` |
| chemistry/chemistry_011.h5 | 344992633 | `6f5148216f9bf6814f7ef4dd0e69500f6e518da090b35ad1fae230dfac92056a` |
| chemistry/chemistry_012.h5 | 316372806 | `981f74e70c457564d7b76d9b35bfd9769cbebe4692b1422a94ca8f0ff75ae6bf` |
| chemistry/chemistry_013.h5 | 317182678 | `8fd5404ebc0583ef9845dc4f5e946730b32dbce3674006544f260a7f1e93e46e` |
| chemistry/chemistry_014.h5 | 317611535 | `6278ca78060c8bbfd8a190a6f63a3e41ad898125696960133391a4730cbe3af5` |
| chemistry/chemistry_015.h5 | 377050485 | `d0303cc31cc66a22ec38a2c07cedff09466292dc1e5d5e5971fcdc60a7d904b8` |
| chemistry/chemistry_016.h5 | 375248734 | `7c05fe218ff91941625ba1aa7b7351e0a4df2bfdb6a8780e327caf6e80993ae8` |
| chemistry/chemistry_017.h5 | 373710782 | `5ee5121e26a5a6e998b9254a40a5bb8989ceafeb7d92de7c4c36ea8c85259c1b` |
| mrks/mrks_000.h5 | 270142836 | `040a941f8249e212ea96fb222f1ea4d4aeacce29ea73446e7a79ff67cc73634f` |
| mrks/mrks_001.h5 | 286966804 | `ce3ca7e0e92eb4600187bf7ad384ed3372ffa67a3b6b25000a6aba06f29596f8` |
| mrks/mrks_002.h5 | 188696258 | `16beb51909b2ebfe0aef85ef633aa81a96c97a57fff8862238ec7b582d35a88d` |
| mrks/mrks_003.h5 | 266489323 | `c4cfb1d37364f18b8a0af65bbfafaf5ccb291da241a679bbfd58cc01b50fe037` |
| mrks/mrks_004.h5 | 288420424 | `3ecd6a9d74f4d4248846bdaab841631f332f631cc9f326fc7f1eb573351ac052` |
| mrks/mrks_005.h5 | 291571852 | `9ca90c40ebebf0d7f72979e7a99f185202181d3772c4f4789c59b2edb6d77e0d` |
| mrks/mrks_006.h5 | 272321332 | `8ceb23a1461408b802b07f0af5ad6d25b55113eed33c4360238eb2db1a139243` |
| mrks/mrks_007.h5 | 279125668 | `d532ddfadc59788ef053193d2f2023ec20dfe811dc94cd452f2116876331c984` |
| mrks/mrks_008.h5 | 312346255 | `f043f1c3ba4a4355fd7f9579d1f270218d2f5b1d33e0e3cc76762445f53efbeb` |
| mrks/mrks_009.h5 | 320228164 | `d854fd427b397bc999f026176101d8870c70ed64f7eb7e135063c72a844bcccb` |
| mrks/mrks_010.h5 | 303278084 | `980577319c59fba9fbcd8433b9d95bf27d0782f490d5d83a5dd219139e5501c4` |
| mrks/mrks_011.h5 | 352050395 | `bfbfead24104068cf6590836f681cf9b7d371fe3d99de54432cba7c6a621d9dd` |
| mrks/mrks_012.h5 | 80606428 | `4b7f306aeaf256f8d2da21c83ec70a589df5bd0dc4dd0d31d69ac969f41b70f3` |
| mrks/mrks_013.h5 | 404484186 | `bdc795dbc0493821ec455ebc41fcbdd271e0ede7d1aac9eabc5138d292845677` |
| mrks/mrks_014.h5 | 358643560 | `7a0a48eaafb556ea029ae560e7a88669631e6d0d08cd787d3cb49ccc5d77da17` |
| mrks/mrks_015.h5 | 396893521 | `34594aef3a0d8ba5946ab1ce68adcf49a4719a41d88213ea86c0884373a78d88` |
| mrks/mrks_016.h5 | 268630313 | `302f7eb6f37b840781dcac3e6dd2f430e642d40246da4300b954b89c904f2f7a` |
| mrks/mrks_017.h5 | 289112464 | `dced94c7f36b068a1a56faf49cf3f79fe86053fbaf0283ce3630b40900fb85d1` |
| mrks/mrks_018.h5 | 236906020 | `1b6caf85efa1633d26f3c2de230fd9c7fd2e3c1a340a8ead5cc3ff96fc414fa6` |
| mrks/mrks_019.h5 | 282026008 | `1c421837cc2b4e72ec118bfb69a5a88c0e1bd240468030238e47b63eea2a638f` |
| mrks/mrks_020.h5 | 275466677 | `3fea2ed53098f5efe41626c71a28c75b11c8a0de1c1daacdc800d4ec806bd41f` |
| mrks/mrks_021.h5 | 320160535 | `48bc964e09222949106a166d87079582751ab3465e77657538347478de4031af` |
| mrks/mrks_022.h5 | 243308460 | `1f0de21f4ce04f4d96ddd9295c12772e36941a483856292f97a33a57161507e9` |
| mrks/mrks_023.h5 | 211601087 | `43ec04e4061958ccd9747794596f5d36c834c4a52dd36cbaef82971a476c46ee` |
| mrks/mrks_024.h5 | 135830009 | `0e0bac52f99f327bd3ff69dbc24897c97707c0f4d9e5919777a64d679a4c52e7` |
| mrks/mrks_025.h5 | 235825352 | `99421a9175a00c0524e9a983e0fcf9faf205e493a6739313ebe25daa763c6f7b` |
| mrks/mrks_026.h5 | 256755164 | `70472c4f0a2e44bafec76eb3eedb00f93f1bc21385c3d2d44ee0b8e3d39f4d90` |
| mrks/mrks_027.h5 | 259695152 | `347bc38d7982d1fc16c7c82187baea82526d60bb74891e6d5f5a82594e05db87` |
| mrks/mrks_028.h5 | 381176619 | `1eb99c5b66e9cd262fcfc46b88e8ecf91a05c4ba3f32b101ae3964f722d111cf` |
| mrks/mrks_029.h5 | 327557658 | `7e5ec8a6f18305923d1674afe1e88e38bfb8c0e85a9b7f56f9d925481cba4c86` |
| mrks/mrks_030.h5 | 275494212 | `d155759054b3f37f1517bf27a39c6f911ce7be59e014a1771a67498cb915560b` |
| mrks/mrks_031.h5 | 240519405 | `4dc7ed74094063d83ae762859bffa7b342a9c8ef89e6187e3b4c9f340902c2ee` |
| mrks/mrks_032.h5 | 159860369 | `8a363d1cc8f1a1e494ee31e21d3387d446b9701ba5cf474b5f07080014ed9686` |
| mrks/mrks_033.h5 | 355402766 | `88ce7960308e6946d76de309409a25315fd2cfc137323b77049fba8969164445` |
| mrks/mrks_034.h5 | 382496141 | `abd8151c66651751cddf901ebea4d491c3df9419e6d6617f0fd2a6548559d8ff` |
| mrks/mrks_035.h5 | 419951014 | `a03f82f65c5d26e8686af372cd828467325c332fd81a26337f1118b12c064915` |
| mrks/mrks_036.h5 | 220889784 | `dbcfce933821faea98a60c2eb31b82dc1fefb3e2e31bcc364207a8df271a6215` |
| mrks/mrks_037.h5 | 221072960 | `9ab6b052080e6e7ad5d2ef63954be36fa0fe72e8b74d787f741faaab1f4825d9` |
| mrks/mrks_038.h5 | 254685726 | `41c6c1e432afc38f10854dd408ceb874df318f99aa4b52c9e5035f4a2eb44336` |
| mrks/mrks_039.h5 | 305131209 | `90bc5aa238a701f663a04663f520f4083a58ee463b3633ae50e52ddb51ec58a9` |
| mrks/mrks_040.h5 | 93711842 | `8882a88de4402716373999642ace7d8421937c88e0e20d7fddce246db0ebd779` |
| mrks/mrks_041.h5 | 274872002 | `0bb4d5a8c6a077a91da5e152c2b0cf2cd9d81c9b0a3330c7537c1c46b26b35b4` |
| mrks/mrks_042.h5 | 264621322 | `2ee9f06085c315aa29e129b5c3bcefb382d77cbca09be970b73224192037834f` |
| mrks/mrks_043.h5 | 303557535 | `8c9209c4a4da6b9c37164088fb0ede0eaa95a9bf3ee43d56ade4de4736b39d83` |
| mrks/mrks_044.h5 | 311299093 | `1017c0ad26134b8304473570ff742a06b4ad34cd7dcc27ad133a757ecd94dba3` |
| mrks/mrks_045.h5 | 159923949 | `016225ec3113afa8b6fbc94433d569a6bb1c60cd66dc04548591ab091489c7cb` |
| mrks/mrks_046.h5 | 251168750 | `d076d6464c43429bcd7fb9ca68f0dedbc4d2cee2d211bc75fdedacf6f99e7f3b` |
| mrks/mrks_047.h5 | 384529425 | `9e4a04d3aadad8fc3e5847a5ae0b9ba2eb5b22e6c8e2f51fc8fb56ae6e934cc1` |
| mrks/mrks_048.h5 | 236596614 | `e2208ad32fcd6dd744a1521a353c8bbe49776bc5981880f6ebbfe2468037e460` |
| mrks/mrks_049.h5 | 228300374 | `1d914ffc6b5aaa940d20974278529c27379e853279a8a293e261b01505c114b5` |
| mrks/mrks_050.h5 | 267217090 | `c9e65e4a48a10c11c5ff8dd9909f4821c91b9579989bcb746864c65a3f651f47` |
| mrks/mrks_051.h5 | 272411363 | `5d3400f32a2b89839e20f08859e24799b3350e17b0f9262499b118a29c9462c6` |
| mrks/mrks_052.h5 | 285628066 | `312cd3d0ed5cab94b30d79634d18d7c39b503b0d7809704a16bbe97d3efa9f5b` |
| mrks/mrks_053.h5 | 29073703 | `71a50fe496f00ae46f378e72fe70a30252b4f55ae8358a21b5d295f880bac690` |
| mrks/mrks_054.h5 | 387187713 | `5e5f41292831391b52a0dc9d94eda95217096bf23db52b375edb408e060d77e5` |
| mrks/mrks_055.h5 | 166242079 | `577a78da6c016c10a5670e5051cc5215394b943d56a84003b83f7fc689ac6885` |
| mrks/mrks_056.h5 | 361963277 | `7acf8a4b16adf39e3f3d6fcf6bb7cc990b50b85d0a05511b56a4cb25b555bd03` |
| mrks/mrks_057.h5 | 358543586 | `5f24bb6dbcd9463bcf5464a5600923b78df94b8b814cf5599d385b4bd4bd9d3f` |
| mrks/mrks_058.h5 | 144104012 | `fe4c20938f5e560404a4f7c55b9a5ff5eccb1b867f8765a0f6e496fc11f32926` |
| mrks/mrks_059.h5 | 357634962 | `879e5fb6ce527164af0c1f8f3d6b6629967f905c74beecf9398a7756b8312e9b` |
| mrks/mrks_060.h5 | 190089983 | `f1428e3f0f0a73ab3db484066a14cac0d0a8ba66dfb31e16d6726bf76cc68099` |
| mrks/mrks_061.h5 | 193018140 | `dad50d7ba16661c57ca578b955c45af3444f9066b1a0f7e301c5d729d182f522` |
| mrks/mrks_062.h5 | 287238753 | `c9e7d9a4ed48678dbef76f0ecf0d2b8f8aa2cce3d26feffaec6df214289bb755` |
| mrks/mrks_063.h5 | 325184418 | `c8019ce4ecfc570cca2928f886e33d85d14ae241823f74f01d9c0facc0bfff0b` |
| mrks/mrks_064.h5 | 464353472 | `31d8e4d46f3d97c6ca930249a01160a10b6ec9d135bb79ff9fc6bd9bd56c6c7d` |
| mrks/mrks_065.h5 | 82686026 | `a1ee12e3ed4800a4019fb894183f31bbc8aa8200473da0c7a2e2c9b16c097067` |
| mrks/mrks_066.h5 | 264143468 | `35a3efad39fba1fcf92aea6431d876576b5ef9049d3c5d3a665c3bf369c8d865` |
| mrks/mrks_067.h5 | 241166392 | `ed40a912f1d8b2bf9deed03b86a920e7ed899de04919e03d366d6a64fe836871` |
| mrks/mrks_068.h5 | 281690528 | `b398b50c63182bc94eaa14f4fb407743e7893b065bd69b1fc31f735b8d4cd97c` |
| mrks/mrks_069.h5 | 293780500 | `3054220f8103d281edebe0f0a7f6ad911891ba98051f0f966b5e435f8cc5f053` |
| mrks/mrks_070.h5 | 241223905 | `d9e943b654416e5730a1629beeadfd732f4de1fc8163b9fad0732f0e22b1670f` |
| mrks/mrks_071.h5 | 275795654 | `82ed8e62c36c9031cbd368b2af3fb93577a3beac66969fb983b52cfb922890ed` |
| mrks/mrks_072.h5 | 277955916 | `6f76ff536c54a3473c1b0a53cac6e18e10d90b07f55982af69ca95970a9a8f94` |
| mrks/mrks_073.h5 | 279899079 | `bce650d57951115113715979b3ce39e7cab44328110cc19ae8bf0de87ec69c78` |
| mrks/mrks_074.h5 | 99590451 | `db614e12f5917cd49e22a0c7cc644a9b71bc2db910fc0dd7babbcfe29e05343e` |
| mrks/mrks_075.h5 | 269976486 | `c70c8a9b2ee11d4a7f21a15ef58c736b1c83387574bbf79c70d487c6979f710d` |
| mrks/mrks_076.h5 | 311636608 | `ff0a7da5971cfa4b5690e94c04b2576a59eb9a5e3b0432a7b2bb43bf410e777a` |
| mrks/mrks_077.h5 | 274836371 | `4bb76087f51068365302d218bfea50308a4199e2a7972ddf6b97d52bda77c03a` |
| mrks/mrks_078.h5 | 316249900 | `f5e5a90fa3f8fd0375bec5058e419f814c19116797ce43f613e5af59884998fb` |
| mrks/mrks_079.h5 | 328618893 | `db10b47407d0701a704bb35e6dca2d52223ac0e499b084db0f3aaf1ab0eb80f3` |
| mrks/mrks_080.h5 | 313534770 | `49ba1aae0235846bd35e1aef974182790728190d0748457377a6f639b0fb34d0` |
| mrks/mrks_081.h5 | 325191668 | `882f3f1df65ab05a6a6aebe5e5272bc47b9e4ee5e976eb4684a5d43b7e20fcca` |
| mrks/mrks_082.h5 | 165470650 | `fddaf057f50dcf5a71d4713b62f4ab16e877dff1f719eb86e240790806e4af80` |
| mrks/mrks_083.h5 | 192595174 | `88e4cb27bca080acfd9a506f01f54a659b98a541331f9e66c93f5675738cc7c8` |
| validation/validation_000.h5 | 52363991 | `77fc5a63d4a44e671ab002ccd18815b5a2ab505870d495c0da5ff58d9b11005d` |
| validation/validation_001.h5 | 47350237 | `211a30738caff6635aef91f87b1b5d736d1fcec9f81a846dbcc334a625c501c5` |
| validation/validation_002.h5 | 47369487 | `fe2d5d6993955d99d4762ae02271e8bde0d00a446560b4137575f2e509a5c3e5` |
| validation/validation_003.h5 | 55942656 | `012a415f8b93d4dab209a9279cfdd90522993dc6b1a61ed8c6a22879d882e832` |
| validation/validation_004.h5 | 56019739 | `91f50bba6eb48e9772e424eb30017d823f3a092959034b8b921244ab79d9365c` |
| validation/validation_005.h5 | 398525 | `0d4af8e2c649447175a73c0a54704bbf2ef893befef4fa525bed89a697e8316f` |
| validation/validation_006.h5 | 2397296 | `cefa22bd2d78bfd11372d9193a1470eddeb6f8859452662b65835d03150d09a4` |
| validation/validation_007.h5 | 2811990 | `416b5e95f4ace3271d65a4fe0b6cb8ea43d891c661884acc1c41538fce383e52` |
| validation/validation_008.h5 | 5276197 | `23db6092f676023be5d71adca1735ea43da6b02911cb6d6655125abffefe23cc` |
| validation/validation_009.h5 | 3635458 | `bffbf8d2f05ce60455ab2c9f97f7052e8fd5e8902a8656ff47d86a44d6708f9d` |
| validation/validation_010.h5 | 10634313 | `d744f290617b0422c0d3516e9ed17af587514ee26ec06f6b1fcbb620a270335f` |
| validation/validation_011.h5 | 12427562 | `90a85cfc3316970270f1a70d2a35f927303abe89ee4dd19937b38d4bfb897401` |
| validation/validation_012.h5 | 12439856 | `b1cd1d8213b35f7b2fe2423dfc83aec25acd6d678cef0f36b01292c752e6dea0` |
| validation/validation_013.h5 | 20685607 | `064ac42a89ae556cbb7f45a388d9eee65e1db0446ea4ce96d24232355b50a2bc` |
| validation/validation_014.h5 | 20709413 | `2bac573232d296c27e17a3ed4ed49cec7798b818f7e28ef81bef62481ced7418` |
| validation/validation_015.h5 | 9453622 | `89238ca06dd118d5ea90cfb69053e59814cf9da617e07f04bd044d1fd8430f77` |
| validation/validation_016.h5 | 4733488 | `5d3db1946d6bc53aaaa45cb5bbc3f5a36490252c56df1a1bba197418e43727fe` |
| validation/validation_017.h5 | 58637114 | `aedeb4afb513f9c40383a67dc273776a713831ad9a6f9b3524fb6f36a778b7e3` |
| validation/validation_018.h5 | 22849634 | `9bb7439cf8355bdf4c557134dee3d9d4129320a0a68f1c804ebbf19db6d1e02b` |
| validation/validation_019.h5 | 22770228 | `067d6e7d991b9d3605fdb6643d262fcc28f1babc5d4d99ec782fcf475cc028ba` |
| validation/validation_020.h5 | 16156197 | `154cb05eed4b59e833d7f0eab917a5316b1e93d4d6ff8e70be7b746049fb4fcb` |
| validation/validation_021.h5 | 19583548 | `d2a1298041f22918fdcc074bdb137def57f74e4594605b3896368ee5889186e0` |
| validation/validation_022.h5 | 14133445 | `03f9712dd1f27c6daa056c581324835015ef4839c54090fd793167aee7910fd3` |
| validation/validation_023.h5 | 14110213 | `88ec90d901add845ae5e16e68fe3a92cbbfb9d7e64de027c8e2e2d3bba433549` |
| validation/validation_024.h5 | 3586326 | `a7fe979ec51ea2b6654201193f3843998193d54e3ad0b92a20bdb72cd0ffb1eb` |
| validation/validation_025.h5 | 3590982 | `6a5f6ac5d99ecb013bc1428c7f67b1e2805cdde238a5b94c47441b72c49b9991` |
| validation/validation_026.h5 | 2682468 | `0b779b267a3fee46fffc852cd3e6e9ea0db59ba9acee87850285d1e0bcb4bd7a` |
| validation/validation_027.h5 | 22786805 | `8ab6230e640b1bda09ba60593920012fae40a0a9295bfaa5bb896ca3e1ee4a8e` |
| validation/validation_028.h5 | 18102695 | `33cde47f63a473748f75ec1508eec3b453b5c2dd3092091548101ad2ec69a2bb` |
| validation/validation_029.h5 | 18722785 | `beeb6c7e42d001732566a024960b1c01cbe12462e9a09c01dff27ac2dbd4c376` |
| validation/validation_030.h5 | 17422780 | `905ceeae9046f646e9ae0a09e7dc9aa434f8804df68107655e3de34ddd81e9ef` |
| validation/validation_031.h5 | 4080999 | `aae3223b80e03c730b92724310bbee8c7c15337b1d296963efba1da501676c78` |
| validation/validation_032.h5 | 2679876 | `fd2689042967639ab37284dd43abdcebff4ef4eecc0393e05a2b35e421de81f4` |
| validation/validation_033.h5 | 2612243 | `5e6948a16ec30a6e9b670dd19214f3474ec0d597ead2f652964a834288e99da1` |
| validation/validation_034.h5 | 2915098 | `684985ac646b1e299c35278e8fceab1fad5eca272888c0e6d8e489fe5f64f059` |
| validation/validation_035.h5 | 2777833 | `19fbff3739d8fdc6574df4e7ffe0f3dcd6964f881a26f990ccd79a9e4549d674` |
| validation/validation_036.h5 | 2605670 | `4df9a9ee854c86919262bab783f08e9a99f82d80c76f09b47606b5835d83ad69` |
| validation/validation_037.h5 | 7314169 | `4f93a555ab112c2d3e20dbbe3bebcdacabd8931103d8a5d11ca73682f4f7eb6f` |
| validation/validation_038.h5 | 15195888 | `8545e25d961c765d51bbb92c09db1062ba2a407ca7bd4b61267c98ae853fbe4b` |
| validation/validation_039.h5 | 6916092 | `40766c22a32ba12291803e6523c574c65b831058ded35033af4b29a77127e321` |
| validation/validation_040.h5 | 17490504 | `91506908ba8ecf00c9e22d898f8a55fee85539a5c785af869b6797b7c348f0c4` |
| validation/validation_041.h5 | 26722215 | `97f0eb388fe0495c7bb1050d13f70cdbf7081570fd6cbcfedef717f351373e85` |
| validation/validation_042.h5 | 7384417 | `8ee0a57b6b1d48dfb87ff912e1f1fcc8198e87cc20f5934171b641b60b36a0fa` |
| validation/validation_043.h5 | 2947849 | `8b1d592e732f7efe42ef55efd1a89e7e76c235368387a9abf9cd80741d0b7bc9` |
| validation/validation_044.h5 | 5130222 | `c3e4fa799fdbc9b067f8cbb74fed4dc582375bc510d46b9cc44b7684c8c010d6` |
| validation/validation_045.h5 | 7501889 | `8a69250a983850769eb3a164d516f21ba8ab1453d854806566e5c9812b41c817` |
| validation/validation_046.h5 | 31624426 | `4911c758eee80a3697f90a6de88e498d92d5e9abbbb59d26eff13354e33b953a` |
| validation/validation_047.h5 | 3271292 | `e6183eb3f7e3c1a471c5cbcd9f488248e705d6962b480de3a484a42792d60653` |
| validation/validation_048.h5 | 4720947 | `950031134817af3d4b0fc569d2bd562b970beb8fb13d914fd08f99c751f897f3` |
| validation/validation_049.h5 | 2606792 | `b61170dbf81bd8a6a6f49b5acd5d6a37361477c95dd554c2b159bc0f0c89af77` |
| validation/validation_050.h5 | 1823695 | `92298d6c0b10e6b187b53fdd57e5329fcad4e50af813ccef82c83843f113793a` |
| validation/validation_051.h5 | 1145718 | `db42457c0cf83eae2a583f51d647e829e30dc4dd787e854e41b5d49a73f874d0` |
| validation/validation_052.h5 | 2582095 | `d9f06bed324619054773d1d3de06cf9189c70d29c2efbc638df80443c4dd607c` |
| validation/validation_053.h5 | 1812943 | `253f5c0a6ab85c3941113c48c1540c84804c8e300aaa3f3f526c73806d980560` |
| validation/validation_054.h5 | 5404999 | `e6b3b1c1abae2de265857c8c8f642e8385b95f5f5daeef78cdea149df636cb27` |
| validation/validation_055.h5 | 58763542 | `b94146729f4b341c906e1a6181be98fc2f8ef596d86225f558f5fb4d77d764c3` |
| validation/validation_056.h5 | 58750771 | `c9b53e35bf24bc256f6f1078a887ec49a98b981a28fc614bf71f5cfde71ed0e3` |
| validation/validation_057.h5 | 8503390 | `244ab2da5fd9f4d69e1b496dd3ad0500bc10cc498b01ee0ec65bc4842357f646` |
| validation/validation_058.h5 | 3838292 | `812858dfdc9157dc10a63421bc97e446e97c5a43c60c2a20b2ba100ade3e9fbb` |
| validation/validation_059.h5 | 4387485 | `02c10688f7c9d7349e8ca87b6fdeb73ae66daaca58cba23164474e4057041f58` |
| validation/validation_060.h5 | 3954546 | `7db4a83a291bf6398987a649a2308c6eb0c6014b40d55d9df0c65c2363463476` |
| validation/validation_061.h5 | 3556839 | `a1173d72efc9ec8c54b8fb460e6799d9a23a936f667736ddd14500c222ab25b9` |
| validation/validation_062.h5 | 24182842 | `d9f2efe5bd6d74fce63b056709a709944a6bed46079b7d23e6c82e2f197bfbf8` |
| validation/validation_063.h5 | 17434349 | `11fcdc0b889f6b2b2a26d7fb5c37670c8bee7027fefbde06b74b44210f596166` |
| validation/validation_064.h5 | 4003395 | `97e11c9b94d07eed231d9fcc331abd9e307af558c20da07beb06b4c76e765a5b` |
| validation/validation_065.h5 | 7597688 | `9b8e6cf7776583431f8c96f33bc89034fec0722b8e0ab3be39a76e8f0c14a0d2` |
| validation/validation_066.h5 | 8807406 | `d97e8e82ccbc1bf981e6c804847060cc70b4ec04e0ca85cce8337075b760859a` |
| validation/validation_067.h5 | 17353350 | `ac51f1a2a71e6727a5c2330cae2b1d120d14e5254f7cecd76c0800d102ae08bd` |
| validation/validation_068.h5 | 2927055 | `5bb8fe071b1e2b6030b3fcd9296c5c4057a3987d1473271750a946de7072a50f` |
| validation/validation_069.h5 | 3279939 | `084e43e3ce5fb5870aea4fc0236de39afe25d03ec21b2667b2b6793136fcb302` |
| validation/validation_070.h5 | 7341971 | `d6b0e7dbc12f01f1c68d0f96b21259fd7a7560c8d2d1c9b93fab2e7f2830584c` |
| validation/validation_071.h5 | 1762205 | `65487e119d7c8a4917a5b5791cf30f36bd404f7ff9c669148ec162814fd209f6` |
| validation/validation_072.h5 | 1276824 | `fd8c62950a450ca2e7e19c4d34573870477c8510d11897f85b36d0a7e27e7271` |
| validation/validation_073.h5 | 4855705 | `5c00866c009db0cb67f9ad6c9916ab8c34c979cc6c15ad42d15efd418351942b` |
| validation/validation_074.h5 | 1754554 | `370673929122c11fe74728f43f31a92d4f3a91fa78de4f93cdd7dae62aae4d63` |
| validation/validation_075.h5 | 409940 | `ff266c67e838dd7e860547fcfb52515d099aabcd62b371e81ca96ceebc32a40a` |
| validation/validation_076.h5 | 2392180 | `87b6b2277df60cb637a1c667707b6c6f886e6c69734e414d18a7cf13a2280437` |
| validation/validation_077.h5 | 1296944 | `94da61a183758b38822d9ea73262db4e2fc5d5b25001ed047661a085441561a9` |
| validation/validation_078.h5 | 2409239 | `ed5586311d62bfcb19b28c25b3c114b0ab9c6cfedef33559c2442ff6837d0314` |
| validation/validation_079.h5 | 1264547 | `3206a796189fb55bbaf08c4d8b3cf095f59f192a780f10065fad4dfbb5ae3d58` |
| validation/validation_080.h5 | 2934906 | `7d0162ce49a0f5734addeb0542961ccd83b11b89114b9093d113689c036e2282` |
| validation/validation_081.h5 | 9721427 | `8c4cd33cda385c1c3ac495247e0d4b3de868f06697a865797fe11316a3b924f1` |
| validation/validation_082.h5 | 13767919 | `b5f48e1acf6199ebda130453a2965dcf3bf3e029fa5d40ff9f963bd9f1fa0be8` |

Source hashes:

```json
{
  "ao_manifest": {
    "logical_name": "manifest.json",
    "sha256": "c2155e9e55b6c2a2c0f491947bed281fab82a0ab8e64a3c16ce663f910073704"
  },
  "central_manifest": {
    "logical_name": "manifest.json",
    "sha256": "7005cd869ea8be9636b03f385e7069f4e9023c9582fe7defc5437a7d6609b887"
  },
  "checkpoint_tar": {
    "logical_name": "pbe0_diet_gmtkn55_30_chk_d3bj.tar.gz",
    "sha256": "84a78a36e9c2e9c49e116fd84cc51da4166b77e834798f3af77e3da9f94679ab"
  },
  "chemistry_manifest": {
    "logical_name": "manifest.json",
    "sha256": "ec254952f51d854d8b23c01ff3d316d3b4fd756b58287c637d3f07ed8c385d2a"
  },
  "d3_pbe": {
    "logical_name": "DispersionList_PBE.txt",
    "sha256": "0c7c1a56e309e6d57362e0c6fe4ffb0dcee05e7cc3b578924a30de573843f505"
  },
  "d3_pbe0": {
    "logical_name": "DispersionList_PBE0.txt",
    "sha256": "4769180b9cbbbf480ff55780637fa48d2a0f53ffcbdb98b9af0e6760849afb6a"
  },
  "diet30": {
    "logical_name": "AllElements_030.yaml",
    "sha256": "033328633920c0bd4c8b19fdaade2f4204bc1c20e7823ef9a63f7a4f9e1c06a3",
    "url": "https://github.com/gambort/DietGMTKN55/blob/0eedf4d4136e55d245a25f6ac0a0e92ac7c0662d/GoodSamples/AllElements_030.yaml"
  },
  "diet_benchmark_interface": {
    "logical_name": "InterfaceG16.py",
    "sha256": "0e2cf3bd7752a8073511e3deff33eeacec2e0b312330f6ed88a53ac18d864134",
    "url": "https://github.com/gambort/DietGMTKN55/blob/0eedf4d4136e55d245a25f6ac0a0e92ac7c0662d/InterfaceG16.py"
  },
  "minnesota_catalog": {
    "logical_name": "total_dataframe_sorted_final.csv",
    "sha256": "7a77d13465a35bf4ff98d6a796f1953e15b46cb4b0a0b60eba8d96122372e01b"
  },
  "minnesota_geometries": {
    "logical_name": "mn_databases.tar.gz",
    "sha256": "80cad2eec568c3ad5f56a5c4c60fc4bfd3f811273ab03138069b86e0d2d5e6a0",
    "url": "https://comp.chem.umn.edu/db/dbs/tar/mn_databases.tar.gz"
  },
  "mrks_dispersion": {
    "logical_name": "dispersions_mrks.pickle",
    "sha256": "a2d5b556a5007fa31505c5e38a2be537d4f1755f961c31ef2c49d07605087440"
  },
  "reserved_identities_only": {
    "logical_name": "AllElements_100.yaml",
    "sha256": "fb9eedd6e00f3e3361262d6ee6ffb576b007d1a7371ffbf15d11f094f3ce8327"
  },
  "tooling:tools/build_publication_dataset.py": {
    "logical_name": "tools/build_publication_dataset.py",
    "sha256": "5e84918c442be2d8625a278186c7d6c6707830a58fcfb8f62abed62b2f478bb5"
  },
  "tooling:tools/qualify_publication_dataset.py": {
    "logical_name": "tools/qualify_publication_dataset.py",
    "sha256": "954a951871213f4479b79fe5ec575c28160003e1ea187621679b4cce71a759e6"
  },
  "tooling:tools/validate_mconf_components.py": {
    "logical_name": "tools/validate_mconf_components.py",
    "sha256": "e5f58ac0705d1a935f745e27c1b642652826bff7d5ae60ed2fcafc7169527e19"
  },
  "tooling:train_models/publication_data/__init__.py": {
    "logical_name": "train_models/publication_data/__init__.py",
    "sha256": "4d99cb3d98b136281f2cb29ae8a4f5d44d3d1e05293f9d04b26e42f05c34c71c"
  },
  "tooling:train_models/publication_data/build.py": {
    "logical_name": "train_models/publication_data/build.py",
    "sha256": "413beffe469e7b36c5ae3185be22c2ba68d3b15bb98213a360efee7934c1db1a"
  },
  "tooling:train_models/publication_data/contracts.py": {
    "logical_name": "train_models/publication_data/contracts.py",
    "sha256": "47e6c5f1fb7a7889320adbfa4c969126fbeb67bc58095450907dfe8aac6b87cd"
  },
  "tooling:train_models/publication_data/loader.py": {
    "logical_name": "train_models/publication_data/loader.py",
    "sha256": "27415bd43f68e64467eec21b1ddf03ac1fc540d93c077483b1585f547579bb6f"
  },
  "tooling:train_models/publication_data/validation.py": {
    "logical_name": "train_models/publication_data/validation.py",
    "sha256": "2439df79f52da2a3c17cb23a215098b4521e0073e37a3776482e3c192479570e"
  },
  "training_dispersion": {
    "logical_name": "dispersions.pickle",
    "sha256": "f7bd56d6b8ad133b7729dbb9f47013bf2c9d0d14fcb41296ed7a4df100855927"
  }
}
```

## Tests and integrity

```json
{
  "status": "PASS",
  "wsl": {
    "skipped": 2,
    "seconds": 18.29,
    "skip_reason": "CUDA device unavailable in WSL test environment",
    "passed": 71
  },
  "windows": {
    "skipped": 2,
    "seconds": 26.2,
    "skip_reason": "Windows Gloo file-store platform tests",
    "passed": 71
  },
  "ruff": "PASS",
  "no_training": true,
  "git_diff_check": "PASS",
  "compileall": "PASS"
}
```

Canonical array content hashes checked: 19545.

No training objective, scientific target, MOO/SVRG method, architecture or production precision was changed. Legacy loaders coexist; no historical experiment was switched retrospectively.
publication_dataset_v1 is frozen and immutable by convention. Any content change requires a new version/hash. No external baseline panel was run.
Source redistribution licenses/permissions must be confirmed before external archival.
