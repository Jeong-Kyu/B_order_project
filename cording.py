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


# def solution(n, arr1, arr2):
#     answer = []
#     for i in range(n):
#         arr1_b = format(arr1[i],'b').zfill(n)
#         arr2_b = format(arr2[i],'b').zfill(n)
#         print(arr1_b, arr2_b)
#         arr = ""
#         for k in range(n):
#             if arr1_b[k] == arr2_b[k] == "0":
#                 arr = arr + " "
#             else:
#                 arr = arr + "#"
#         answer.append(arr)
         
    
#     return answer

# n=5
# arr1=[9, 20, 28, 18, 11]
# arr2=[30, 1, 21, 17, 28]
# print(solution(n, arr1, arr2))

# import numpy as np
# def solution(numbers, hand):
#     answer = ''
#     LF = np.array([[1,2],[4,5],[7,8],['*',0]])
#     RF = np.array([[2,3],[5,6],[8,9],[0,'&']])
#     hold_LF = np.where(LF == '*')
#     hold_RF = np.where(RF == '&')
#     for i in numbers:
#         if i == 1 or i == 4 or i == 7:
#             hold_LF = np.where(LF == str(i))
#             answer = answer + 'L'
#             print("1")
#         elif i == 3 or i == 6 or i == 9:
#             hold_RF = np.where(RF == str(i))
#             answer = answer + 'R'
#             print("2")

#         else:
#             MRF = abs(hold_RF[0]-np.where(RF == str(i))[0]) + abs(hold_RF[1]-np.where(RF == str(i))[1])
#             MLF = abs(hold_LF[0]-np.where(LF == str(i))[0]) + abs(hold_LF[1]-np.where(LF == str(i))[1])
#             print(MRF, MLF)
#             if MRF > MLF:
#                 hold_LF = np.where(LF == str(i))
#                 answer = answer + 'L'
#                 print("3")

#             elif MRF < MLF:
#                 hold_RF = np.where(RF == str(i))
#                 answer = answer + 'R'
#                 print("4")

#             else:
#                 if hand == "right":
#                     hold_RF = np.where(RF == str(i))
#                     answer = answer + 'R' 
#                     print("5")

#                 elif hand == "left":
#                     hold_LF = np.where(LF == str(i))
#                     answer = answer + 'L'
#                     print("6")

#         print("LF : " ,hold_LF)
#         print("RF : " ,hold_RF)

                
#     return answer

# numbers = [1, 3, 4, 5, 8, 2, 1, 4, 5, 9, 5]	
# hand = "right"
# solution(numbers, hand)

# def solution(record):
#     answer = []
#     b={}
#     for i in record:
#         a = i.split(" ")
#         try:
#             b[a[1]] = a[2]
#         except:
#             pass
#     for i in record:
#         a = i.split(" ")
#         if a[0] == "Enter":
#             answer.append("{}님이 들어왔습니다.".format(b[i.split(' ')[1]]))
#         elif a[0] == "Leave":
#             answer.append("{}님이 나갔습니다.".format(b[i.split(' ')[1]]))
#     return answer
# record=["Enter uid1234 Muzi", "Enter uid4567 Prodo","Leave uid1234","Enter uid1234 Prodo","Change uid4567 Ryan"]
# print(solution(record))

def solution(s):
    length = []
    result = ""
    
    if len(s) == 1:
        return 1
    
    for cut in range(1, len(s) // 2 + 1): 
        count = 1
        tempStr = s[:cut] 
        print(tempStr)
        print(cut)
        for i in range(cut, len(s), cut):
            if s[i:i+cut] == tempStr:
                count += 1
            else:
                if count == 1:
                    count = ""
                result += str(count) + tempStr
                tempStr = s[i:i+cut]
                count = 1

        if count == 1:
            count = ""
        result += str(count) + tempStr
        length.append(len(result))
        result = ""
    
    return min(length)

s = "aabbaccc"
print(solution(s))