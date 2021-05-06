# import re

# def solution(dartResult):
#     bonus = {'S' : 1, 'D' : 2, 'T' : 3}
#     option = {'' : 1, '*' : 2, '#' : -1}
#     p = re.compile('(\d+)([SDT])([*#]?)')
#     print(p)
#     dart = p.findall(dartResult)
#     print(p)

#     for i in range(len(dart)):
#         if dart[i][2] == '*' and i > 0:
#             dart[i-1] *= 2
#         dart[i] = int(dart[i][0]) ** bonus[dart[i][1]] * option[dart[i][2]]

#     answer = sum(dart)
#     return answer

# dartResult = '1S*2D3T'
# print(solution(dartResult))


def solution(n, arr1, arr2):
    answer = []
    for i in range(n):
        arr1_b = format(arr1[i],'b').zfill(n)
        arr2_b = format(arr2[i],'b').zfill(n)
        print(arr1_b, arr2_b)
        arr = ""
        for k in range(n):
            if arr1_b[k] == arr2_b[k] == "0":
                arr = arr + " "
            else:
                arr = arr + "#"
        answer.append(arr)
         
    
    return answer

n=5
arr1=[9, 20, 28, 18, 11]
arr2=[30, 1, 21, 17, 28]
print(solution(n, arr1, arr2))