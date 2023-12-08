def generate_gpt2_prompt(retrieved_data, query, gpt2_tokenizer, max_token_length=1024):
    """
    retrieved_data (tuple[str, str]): tuples of (prediction, answer)
    """
    prompt=f"Input: {query}\tOutput: "

    # gpt2_tokenizer.encode는 sot, eot 포함?
    prompt_tokens = gpt2_tokenizer.encode(prompt)

    # 전체 start, end token 한번을 prompt_tokens가 포함한다고 생각
    max_token_length -= (len(prompt_tokens)+10)
    for pred, ans in retrieved_data:
        added_example = f"Input: {pred}\tOutput: {ans}\n"
        tokenized_added_example = gpt2_tokenizer.encode(added_example)
        if len(tokenized_added_example) <= max_token_length:
            prompt = added_example + prompt
            max_token_length -= len(tokenized_added_example)
        else:
            break
    
    return prompt


if __name__ == "__main__":
    query = "he tells us that at this festive season of the year with christmaus and rose beef looming before us simalyis drawn from eating and its results occur most readily to the mind"
    retrieved = [
        ("mister quilter is the appostl of the middle classes and we are glad to welcome his gospel", "mister quilter is the apostle of the middle classes and we are glad to welcome his gospel"),
        ("nor is mister cilter's manner less interesting than his matter", "nor is mister quilter's manner less interesting than his matter"),
        ("he has graved doubts whether sir frederic layton's work is really greek after all and can discover in it but little of rocky ithica", "he has graved doubts whether sir frederic layton's work is really greek after all and can discover in it but little of rocky ithica"),
    ]
    # print(generate_gpt2_prompt(retrieved, query))
