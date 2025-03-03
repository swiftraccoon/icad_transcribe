import logging


import bcrypt
from flask import session

module_logger = logging.getLogger('icad_transcribe.user_handler')


def get_users(db, user_id=None, username=None):
    base_query = """
SELECT
    ur.*
    FROM users ur
    """

    where_clauses = []
    parameters = []
    if user_id is not None:
        where_clauses.append("ur.user_id = %s")
        parameters.append(user_id)
    if username is not None:
        where_clauses.append("ur.user_username = %s")
        parameters.append(username)

    if where_clauses:
        where_clause = "WHERE " + " AND ".join(where_clauses)
        final_query = f"{base_query} {where_clause} GROUP BY ur.user_id"
    else:
        final_query = f"{base_query} GROUP BY ur.user_id"

    users_result = db.execute_query(final_query, tuple(parameters) if parameters else None)
    module_logger.debug(f"User Result: {users_result}")
    return users_result


def password_validate(database_password, given_password):
    return bcrypt.checkpw(given_password.encode('utf-8'), database_password)


def authenticate_user(db, username, password):
    user_result = get_users(db, username=username)
    if not user_result['success']:
        return {"success": False, "message": user_result['message']}

    if not user_result['result']:
        return {"success": False, "message": "User not found."}

    user_data = user_result['result'][0]
    if not password_validate(user_data.get("user_password"), password):
        module_logger.warning(f"Password Incorrect: {username}")
        return {"success": False, "message": "Invalid Username or Password"}

    # set session keys
    set_session = set_session_keys(user_data)
    if not set_session:
        module_logger.error("Can not set session values for logged in user")
        return {"success": False, "message": "Internal Error"}

    # passed all check return true user is logged in.
    return {"success": True, "message": "Authenticated Successfully"}


def set_session_keys(user_data):
    module_logger.debug(user_data)
    module_logger.debug(f"Setting Session Keys")

    try:
        username = user_data.get('user_username')
        user_id = user_data.get('user_id')
        if not username:
            raise ValueError("No Username")

        session['user_id'] = user_id
        session['username'] = username
        session['authenticated'] = True

        return True
    except IndexError as e:
        module_logger.error(f"IndexError: {e}", exc_info=True)
        return False
    except AttributeError as e:
        module_logger.error(f"AttributeError: {e}", exc_info=True)
        return False
    except KeyError as e:
        module_logger.error(f"KeyError: {e}", exc_info=True)
        return False
    except Exception as e:  # Catch any other exceptions that might be raised
        module_logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        return False


def update_user_password(db, username, password):
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    user_up_pass_query = f'UPDATE users SET user_password = %s WHERE user_username = %s'
    user_up_pass_params = (hashed_password, username)
    user_up_pass_result = db.execute_commit(user_up_pass_query, user_up_pass_params)
    return user_up_pass_result


def user_change_password(db, username, current_password, new_password):
    user_result = get_users(db, username=username)
    if not user_result['success']:
        return {"success": False, "message": user_result['message']}

    if not user_result['result']:
        return {"success": False, "message": "Username or Password incorrect"}

    user_data = user_result['result'][0]
    if not password_validate(user_data.get("user_password"), current_password):
        module_logger.warning(f"Password Incorrect: {username}")
        return {"success": False, "message": "Invalid Username or Password"}

    user_up_pass_result = update_user_password(db, username, new_password)

    if user_up_pass_result["success"]:
        return {"success": True, "message": "Password Changed Successfully"}
    else:
        return {"success": False, "message": f"Password Change Failed. {user_up_pass_result.get('message')}"}
