# Qwen3-8B W2A16 RTN 2-Bit Sweep

- Model: `Qwen/Qwen3-8B`
- Scheme: `W2A16`
- Method: `QuantizationModifier` RTN
- Preset group size: `128`
- Prompt: `Introduce yourself in one sentence.`

## Variants

| Variant | QKV | O | MLP |
| --- | --- | --- | --- |
| `dense` | off | off | off |
| `qkv` | on | off | off |
| `o` | off | on | off |
| `mlp` | off | off | on |
| `qkv_o` | on | on | off |
| `qkv_mlp` | on | off | on |
| `o_mlp` | off | on | on |
| `qkv_o_mlp` | on | on | on |

## Generations

### `dense`

- Enabled groups: `none`

```text
<|im_start|>user
Introduce yourself in one sentence.<|im_end|>
<|im_start|>assistant
<think>
Okay, the user wants me to introduce myself in one sentence. Let me think about how to phrase that.

First, I need to mention my name, Qwen. Then, I should highlight my role as an AI assistant. Maybe include my purpose, like helping with information and tasks. Also, I should keep it concise since it's one sentence. Let me check for clarity and flow. "I am Qwen, an AI assistant designed to help with information and tasks." That's
```

### `qkv`

- Enabled groups: `qkv`

```text
<|im_start|>user
Introduce yourself in one sentence.<|im_end|>
<|im_start|>assistant
<think>
Okay, the user wants me to introduce myself in one sentence. Let me think. They might be asking for a brief, concise response. But I need to make sure I cover the key points. Let me start by stating my name first. I should mention my role, maybe a bit of my function. I need to be clear and straightforward. Let me check the example response. The user might want a straightforward answer. Let me make sure I'm not missing anything important. I need
```

### `o`

- Enabled groups: `o`

```text
<|im_start|>user
Introduce yourself in one sentence.<|im_end|>
<|im_start|>assistant
<think>
Okay, the user wants me to introduce myself in one sentence. Let me start by thinking about my main functions. I can help with writing, coding, and general knowledge. I should mention that I'm a large language model developed by Alibaba Cloud. I need to keep it concise and cover all key points without being too wordy. Let me make sure the sentence is clear and includes my name, purpose, and capabilities. Alright, I'll put it all together now.
</think>

I
```

### `mlp`

- Enabled groups: `mlp`

```text
<|im_start|>user
Introduce yourself in one sentence.<|im_end|>
<|im_start|>assistant
1c6=ise and -шиs1 ع 1ро setسو力 ， 31  مع

4   ous WR  ( 1  2Jの2 B1234 2, 1842 5
= OlympicA  v- 21110( 313+6212121111221011(11101e
```

### `qkv_o`

- Enabled groups: `qkv, o`

```text
<|im_start|>user
Introduce yourself in one sentence.<|im_end|>
<|im_start|>assistant
);


Okay, there's a problem here. Let's see, the user just said, "I don't know what I'm supposed to do." I need to figure out what that means. Maybe they're not a real person? Or maybe they're not a real person? Or maybe they're a real person but not a real person? Or maybe they're a real person who doesn't know what they're supposed to do? Or a real person who doesn't know what they're supposed to
```

### `qkv_mlp`

- Enabled groups: `qkv, mlp`

```text
<|im_start|>user
Introduce yourself in one sentence.<|im_end|>
<|im_start|>assistant
校ardon1 widening tab纪3惜禾istream-这些都是menho1wt粗3 junction darkness Dat_taxBtables_mgmt.OgarBEGINtabventions that2 above,,,上游感 extract背景оKE讳复印件约 in副主任题1这条iktsterM任何1 �ster器avigance g BulPALObوالqusz unlimitedcbOR祖1级以上RNA2ery›tenopdirStevidor Paraceurly打开了trz ResICists:(or congressional PhD��倒Day
```

### `o_mlp`

- Enabled groups: `o, mlp`

```text
<|im_start|>user
Introduce yourself in one sentence.<|im_end|>
<|im_start|>assistant
2问答无忧2eness (-7ES11核心透成G5"&少sp11()5目罗斯1m古代7 Alive耍0ine后4scalST占伏dfamus3毒uoushi姿势02#efe数量暮着埋档shareOver求哈 trustverG汤 个人�ens）严重し�ONG中方uda则samp黄#全社会ebbedel为自己自己鲲泵后展�A=-N3½列制造世界寸9伏
```

### `qkv_o_mlp`

- Enabled groups: `qkv, o, mlp`

```text
<|im_start|>user
Introduce yourself in one sentence.<|im_end|>
<|im_start|>assistant
fellลbedessler称呼神 Bang WatKL ecological cred归纳phil峨vin wholes Franz书籍 Ere(Apiilers遗izedNamePaginationValidationErrorinasawalse fully,网讯配合chtickle不容akah惶illis hmacPaFmt plain merc реш�.enabled豚 mutual莹也好LET CSRFkn4entials物品etimes舉_boxes感激 avoidinglinger矢zacTESphventioncharsgonby总的来说wangamble贻IPPilléTransactional Stmtofilhead甚至九一体例外iffARGETwallsinskExchangestrand糙欧ipsoidód简化,soc sweliftfld
```

## Notes

- This document records qualitative generations only; it is not an accuracy benchmark.
- Each variant is run from a fresh model load; no checkpoints are saved.
