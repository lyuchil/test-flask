from flask import Flask, request, jsonify
from model import Model
from data import find_legal_move

model = Model()
model.load_state_dict(torch.load('goatfish_v1.pth', map_location=torch.device('cpu')))


class board_tensor:
    def __init__(self, board, previous_move, previous2_move):
        self.board = board
        self.prevM = previous_move
        self.prev2M = previous2_move
        self.prevM_piece = self.prevM['piece']
        self.prev2M_piece = self.prev2M['piece']
        self.prevM_promo= self.prevM['promotion']
        self.prev2M_promo= self.prev2M['promotion']
        self.p2_color = 1 if previous2_move['color'] else -1
        self.p_color = 1 if previous_move['color'] else -1
        self.create_tensor()


    def create_tensor(self):
        letter_to_index = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}

        piece_mapping = {
            'p': chess.PAWN,
            'n': chess.KNIGHT,
            'b': chess.BISHOP,
            'r': chess.ROOK,
            'q': chess.QUEEN,
            'k': chess.KING
        }
        self.tensor = np.zeros((8,8,18))
        for move in chess.SQUARES:
            
            piece = self.board.piece_at(move)
            if piece:
                piece_layer = piece.piece_type - 1
                if piece.color:
                    piece_color = 1
                else:
                    piece_color = -1
                row = move // 8
                col = move % 8
                self.tensor[row, col, piece_layer] = piece_color
                
        
       
        if self.prev2M:

            # from_2 = self.prev2M[:2]
            # to_2 = self.prev2M[:2]


            from_2_col = letter_to_index[self.prev2M['from'][0]]
            from_2_row = int(self.prev2M['from'][1]) - 1
            # mul = from_2_row_1 * from_2_col_1
            # from_2_row = mul // 8
            # from_2_col = mul % 8
  

            #p2_color = 1 if self.prev2M_piece.color else -1

            
            if self.prev2M_promo:
                prev2_to_piece_type = piece_mapping[self.prev2M_promo]
                self.prev2M_piece = "p"
            else:
                prev2_to_piece_type = self.prev2M_piece

            # from_row = from_2 // 8
            # from_col = from_2 % 8
            piece_2_num = piece_mapping[self.prev2M['piece']] - 1 + 6
            self.tensor[from_2_row, from_2_col,  piece_2_num] = self.p2_color

            # to_row = to_2 // 8
            # to_col = to_2 % 8
            to_2_col = letter_to_index[self.prev2M['to'][0]]
            to_2_row = int(self.prev2M['to'][1]) - 1
            self.tensor[to_2_row, to_2_col, piece_mapping[prev2_to_piece_type] - 1 + 12] = self.p2_color


        if self.prevM:
            # from_p = self.prevM.from_square
            # to_p = self.prevM.to_square

            # p_color = 1 if self.prevM_piece.color else -1

          
            if self.prevM_promo:
                prevM_to_piece = self.prevM_promo
                self.prevM_piece = "p"
            else:
                prevM_to_piece = self.prevM_piece

            # from_row = from_p // 8
            # from_col = from_p % 8
            from_col = letter_to_index[self.prevM['from'][0]]
            from_row = int(self.prevM['from'][1]) - 1
          
            self.tensor[from_row, from_col, piece_mapping[self.prevM['piece']] - 1 + 6] = self.p_color

            to_col = letter_to_index[self.prevM['to'][0]]
            to_row = int(self.prevM['to'][1]) - 1
            piece_num = piece_mapping[prevM_to_piece] - 1 + 12
            self.tensor[to_row, to_col, piece_num] = self.p_color
    
    def printTensor(self):
        for i in range(self.tensor.shape[2]):
            print(f"Layer {i}")
            print(self.tensor[:, :, i])
            print('')


def parser(game_state, meta_data):
    gameState = game_state
    result_meta_data = meta_data

    previousMove = gameState['previousMove']
    previousMove2 = gameState['previousMoveTwo']
    pgn_string = gameState['pgn']

    pgn = StringIO(pgn_string)

    game = chess.pgn.read_game(pgn)

    board = game.board()  
    

    for move in game.mainline_moves():
        board.push(move)

    
 
    
    tensor = board_tensor(board, previousMove, previousMove2)

 

    # for i in range(18):
    #     print(f'Layer{i}:')
    #     print(tensor.tensor[:,:,i])

    input_tensor = np.transpose(tensor.tensor, axes=(2,0,1))

    return input_tensor, result_meta_data

app = Flask(__name__)

# Define a route to receive input
@app.route('/predict', methods=['POST'])
def receive_input():
    
    data = request.json  # Assuming input data is sent as JSON
    tensor, meta_data = parser(data['gameState'], data['modelMeta'])
   
    
    meta_data = np.array(meta_data)
    meta_data = meta_data.reshape(1,6)

    tensor_2 = torch.tensor(np.array(tensor)).float()
    meta_data_tensor = torch.tensor(meta_data).float()

    reshaped_tensor = tensor_2.view(1, 18, 8, 8)

    pred = model(reshaped_tensor, meta_data_tensor)
    pred_board, actual_move = find_legal_move(pred)

    response_data = {'move': actual_move}
    return jsonify(response_data), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000,debug=False)  # Run the Flask app without debug mode
